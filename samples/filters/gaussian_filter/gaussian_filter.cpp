/* Copyright (c) 2023, Arm Limited and Contributors
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 the "License";
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "gaussian_filter.h"

GaussianFilter::GaussianFilter()
{
	title = "Gaussian filters collection";
}

GaussianFilter::~GaussianFilter()
{
	if (device)
	{
		for (int i = 0; i < window_count; ++i)
		{
			vkDestroyPipeline(get_device().get_handle(), gaussian_filter_def_pipelines[i], nullptr);

			vkDestroyPipeline(get_device().get_handle(), gaussian_filter_opt_pipelines[i], nullptr);

			vkDestroyPipeline(get_device().get_handle(), gaussian_filter_comp_first_pass_pipelines[i], nullptr);
			vkDestroyPipeline(get_device().get_handle(), gaussian_filter_comp_second_pass_pipelines[i], nullptr);

			vkDestroyPipeline(get_device().get_handle(), gaussian_filter_linear_horiz_pipelines[i], nullptr);
			vkDestroyPipeline(get_device().get_handle(), gaussian_filter_linear_vert_pipelines[i], nullptr);
		}

		vkDestroyPipeline(get_device().get_handle(), resolve_pipeline, nullptr);
		vkDestroyPipeline(get_device().get_handle(), main_pass.pipeline, nullptr);

		vkDestroyRenderPass(get_device().get_handle(), main_pass.render_pass, nullptr);
		vkDestroyRenderPass(get_device().get_handle(), intermediate_filter_pass, nullptr);
		vkDestroyRenderPass(get_device().get_handle(), filter_pass, nullptr);

		vkDestroyFramebuffer(get_device().get_handle(), main_pass.framebuffer, nullptr);
		vkDestroyFramebuffer(get_device().get_handle(), intermediate_filter_pass_framebuffer, nullptr);

		for (int i = 0; i < filter_pass_framebuffers.size(); ++i)
		{
			vkDestroyFramebuffer(get_device().get_handle(), filter_pass_framebuffers[i], nullptr);
		}

		vkDestroyPipelineLayout(get_device().get_handle(), pipeline_layouts.graphics, nullptr);
		vkDestroyPipelineLayout(get_device().get_handle(), pipeline_layouts.compute, nullptr);
		vkDestroyPipelineLayout(get_device().get_handle(), pipeline_layouts.resolve, nullptr);
		
		vkDestroyDescriptorSetLayout(get_device().get_handle(), descriptor_set_layouts.graphics_resolve, nullptr);
		vkDestroyDescriptorSetLayout(get_device().get_handle(), descriptor_set_layouts.compute, nullptr);

		main_pass.texture.image.reset();
		vkDestroySampler(get_device().get_handle(), main_pass.texture.sampler, nullptr);

		vkDestroyQueryPool(get_device().get_handle(), query_pool, nullptr);
	}
}

void GaussianFilter::build_command_buffers()
{
	update_descriptor_sets();
	
	VkCommandBufferBeginInfo command_buffer_begin_info = vkb::initializers::command_buffer_begin_info();

	VkClearValue clear_values;
	clear_values.color        = {{0.0f, 0.0f, 0.0f, 0.0f}};

	VkRenderPassBeginInfo render_pass_begin_info    = vkb::initializers::render_pass_begin_info();
	render_pass_begin_info.renderArea.offset.x      = 0;
	render_pass_begin_info.renderArea.offset.y      = 0;
	render_pass_begin_info.renderArea.extent.width  = width;
	render_pass_begin_info.renderArea.extent.height = height;
	render_pass_begin_info.clearValueCount          = 1;
	render_pass_begin_info.pClearValues             = &clear_values;

	for (int32_t i = 0; i < draw_cmd_buffers.size(); ++i)
	{
		auto cmd = draw_cmd_buffers[i];

		vkBeginCommandBuffer(cmd, &command_buffer_begin_info);

		vkCmdResetQueryPool(cmd, query_pool, 0, 4);

		render_pass_begin_info.renderPass = main_pass.render_pass;
		render_pass_begin_info.framebuffer = main_pass.framebuffer;

		{
			vkCmdBeginRenderPass(cmd, &render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);

			VkViewport viewport = vkb::initializers::viewport(static_cast<float>(width), static_cast<float>(height), 0.0f, 1.0f);
			vkCmdSetViewport(cmd, 0, 1, &viewport);

			VkRect2D scissor = vkb::initializers::rect2D(width, height, 0, 0);
			vkCmdSetScissor(cmd, 0, 1, &scissor);

			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, main_pass.pipeline);

			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
				pipeline_layouts.resolve, 0, 1, &main_pass.set, 0, nullptr);

			vkCmdDraw(cmd, 3, 1, 0, 0);

			vkCmdEndRenderPass(cmd);
		}

		if (type == COMP)
		{
			VkImageMemoryBarrier intermediate_image_barrier;
			intermediate_image_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			intermediate_image_barrier.pNext = nullptr;
			intermediate_image_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			intermediate_image_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			intermediate_image_barrier.srcAccessMask = VK_ACCESS_NONE;
			intermediate_image_barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			intermediate_image_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			intermediate_image_barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
			intermediate_image_barrier.image = storage_intermediate_image->get_handle();
			intermediate_image_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
			
			vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &intermediate_image_barrier);

			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, gaussian_filter_comp_first_pass_pipelines[pipeline_id]);

			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layouts.compute, 0, 1, &descriptor_sets.compute.first, 0, nullptr);
			
			vkCmdPushConstants(cmd, pipeline_layouts.compute, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstCompute), &pushConstCompute);

			uint32_t x_size = width / workgroup_axis_size + (width % workgroup_axis_size != 0);
			uint32_t y_size = height;

			vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool, 0);
			vkCmdDispatch(cmd, x_size, y_size, 1);
			vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool, 1);

			intermediate_image_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			intermediate_image_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			intermediate_image_barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
			intermediate_image_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

			VkImageMemoryBarrier output_image_barrier;
			output_image_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			output_image_barrier.pNext = nullptr;
			output_image_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			output_image_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			output_image_barrier.srcAccessMask = VK_ACCESS_NONE;
			output_image_barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			output_image_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			output_image_barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
			output_image_barrier.image = storage_output_image->get_handle();
			output_image_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

			std::array<VkImageMemoryBarrier, 2> barriers = {intermediate_image_barrier, output_image_barrier};

			vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, barriers.size(), barriers.data());
			
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, gaussian_filter_comp_second_pass_pipelines[pipeline_id]);

			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layouts.compute, 0, 1, &descriptor_sets.compute.second, 0, nullptr);
			
			vkCmdPushConstants(cmd, pipeline_layouts.compute, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstCompute), &pushConstCompute);

			x_size = width;
			y_size = height / workgroup_axis_size + (height % workgroup_axis_size != 0);

			vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool, 2);
			vkCmdDispatch(cmd, x_size, y_size, 1);
			vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool, 3);

			output_image_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			output_image_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			output_image_barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
			output_image_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

			vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 
				0, 0, nullptr, 0, nullptr, 1, &output_image_barrier);
		}

		if (type == LINEAR)
		{
			render_pass_begin_info.framebuffer = intermediate_filter_pass_framebuffer;
			render_pass_begin_info.renderPass = intermediate_filter_pass;

			vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool, 0);
			vkCmdBeginRenderPass(cmd, &render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);

			VkViewport viewport = vkb::initializers::viewport(static_cast<float>(width), static_cast<float>(height), 0.0f, 1.0f);
			vkCmdSetViewport(cmd, 0, 1, &viewport);

			VkRect2D scissor = vkb::initializers::rect2D(width, height, 0, 0);
			vkCmdSetScissor(cmd, 0, 1, &scissor);

			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gaussian_filter_linear_horiz_pipelines[pipeline_id]);

			vkCmdPushConstants(cmd, pipeline_layouts.graphics, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pushConstGraphics), &pushConstGraphics);

			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layouts.graphics, 0, 1, &descriptor_sets.graphics.first, 0, nullptr);
				
			vkCmdDraw(cmd, 3, 1, 0, 0);

			vkCmdEndRenderPass(cmd);
			vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool, 1);
		}

		{
			render_pass_begin_info.framebuffer = filter_pass_framebuffers[i];
			render_pass_begin_info.renderPass = filter_pass;

			if (type != COMP)
				vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool, type == LINEAR ? 2 : 0);
			vkCmdBeginRenderPass(cmd, &render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);

			VkViewport viewport = vkb::initializers::viewport(static_cast<float>(width), static_cast<float>(height), 0.0f, 1.0f);
			vkCmdSetViewport(cmd, 0, 1, &viewport);

			VkRect2D scissor = vkb::initializers::rect2D(width, height, 0, 0);
			vkCmdSetScissor(cmd, 0, 1, &scissor);

			switch (type)
			{
			case DEF:
				vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gaussian_filter_def_pipelines[pipeline_id]);

				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layouts.graphics, 0, 1, &descriptor_sets.graphics.first, 0, nullptr);

				vkCmdPushConstants(cmd, pipeline_layouts.graphics, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pushConstGraphics), &pushConstGraphics);

				vkCmdDraw(cmd, 3, 1, 0, 0);
				break;
			case OPT:
				vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gaussian_filter_opt_pipelines[pipeline_id]);

				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layouts.graphics, 0, 1, &descriptor_sets.graphics.first, 0, nullptr);

				vkCmdPushConstants(cmd, pipeline_layouts.graphics, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pushConstGraphics), &pushConstGraphics);

				vkCmdDraw(cmd, 3, 1, 0, 0);
				break;
			case LINEAR:
				vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gaussian_filter_linear_vert_pipelines[pipeline_id]);

				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layouts.graphics, 0, 1, &descriptor_sets.graphics.second, 0, nullptr);
				
				vkCmdPushConstants(cmd, pipeline_layouts.graphics, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pushConstGraphics), &pushConstGraphics);

				vkCmdDraw(cmd, 3, 1, 0, 0);
				break;
			case COMP:
				vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, resolve_pipeline);

				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layouts.resolve, 0, 1, &descriptor_sets.resolve, 0, nullptr);
				
				vkCmdDraw(cmd, 3, 1, 0, 0);
				break;
			}
			vkCmdEndRenderPass(cmd);

			if (type != COMP)
				vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool, type == LINEAR ? 3 : 1);
		}
		
		{
			render_pass_begin_info.renderPass = render_pass;
			render_pass_begin_info.framebuffer = framebuffers[i];

			vkCmdBeginRenderPass(cmd, &render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);
	
			draw_ui(cmd);

			vkCmdEndRenderPass(cmd);
		}

		VK_CHECK(vkEndCommandBuffer(cmd));
	}
}

void GaussianFilter::render(float delta_time)
{
	if (!prepared)
	{
		return;
	}
	ApiVulkanSample::prepare_frame();
	submit_info.commandBufferCount = 1;
	submit_info.pCommandBuffers    = &draw_cmd_buffers[current_buffer];
	VK_CHECK(vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE));
	ApiVulkanSample::submit_frame();
	get_frame_time();
}

bool GaussianFilter::prepare(const vkb::ApplicationOptions &options)
{
	if (!VulkanSample::prepare(options))
	{
		return false;
	}

	depth_format = vkb::get_suitable_depth_format(device->get_gpu().get_handle());

	VkSemaphoreCreateInfo semaphore_create_info = vkb::initializers::semaphore_create_info();
	VK_CHECK(vkCreateSemaphore(device->get_handle(), &semaphore_create_info, nullptr, &semaphores.acquired_image_ready));
	VK_CHECK(vkCreateSemaphore(device->get_handle(), &semaphore_create_info, nullptr, &semaphores.render_complete));

	submit_info                   = vkb::initializers::submit_info();
	submit_info.pWaitDstStageMask = &submit_pipeline_stages;

	if (window->get_window_mode() != vkb::Window::Mode::Headless)
	{
		submit_info.waitSemaphoreCount   = 1;
		submit_info.pWaitSemaphores      = &semaphores.acquired_image_ready;
		submit_info.signalSemaphoreCount = 1;
		submit_info.pSignalSemaphores    = &semaphores.render_complete;
	}

	// queue = device->get_suitable_graphics_queue().get_handle();

	queue = device->get_queue_by_flags(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT, 0).get_handle();

	uint32_t validBits = device->get_queue_by_flags(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT, 0).get_properties().timestampValidBits;
	assert(validBits);
	LOGI(validBits);
	validBits = sizeof(uint64_t) * CHAR_BIT - validBits;
	mask = 0;
	mask = ~mask >> validBits;
	LOGI("valid bits mask = {0:x}", mask);

	create_swapchain_buffers();
	setup_images();
	create_command_pool();
	create_command_buffers();
	create_synchronization_primitives();
	setup_depth_stencil();
	setup_render_pass();
	create_pipeline_cache();
	setup_framebuffer();

	width  = get_render_context().get_surface_extent().width;
	height = get_render_context().get_surface_extent().height;

	prepare_gui();

	// fill push constants
	{
		pushConstCompute.width = width;
		pushConstCompute.height = height;
		pushConstCompute.gaussian_divisor = -0.5f / (sigma * sigma);

		pushConstGraphics.offset_width = 1.0f / width;
		pushConstGraphics.offset_height = 1.0f / height;
		pushConstGraphics.gaussian_divisor = pushConstCompute.gaussian_divisor;
	}

	main_pass.texture = load_texture(texture_path.data(), vkb::sg::Image::Color);

	setup_query_pool();
	setup_descriptor_set_layouts();
	prepare_pipelines();
	setup_descriptor_pool();
	setup_descriptor_sets();
	update_descriptor_sets();
	build_command_buffers();
	prepared = true;
	return true;
}

void GaussianFilter::on_update_ui_overlay(vkb::Drawer &drawer)
{
	bool reset = false;
	if (drawer.header("Select shader"))
	{
		if (drawer.button("3x3"))
		{
			pipeline_id = 0;
			reset = true;
		}
		ImGui::SameLine();
		if (drawer.button("5x5"))
		{
			pipeline_id = 1;
			reset = true;
		}
		ImGui::SameLine();
		if (drawer.button("7x7"))
		{
			pipeline_id = 2;
			reset = true;
		}

		int32_t curIndex = (type == COMP ? 3 : (type == LINEAR ? 2 : type == OPT));
		if (drawer.combo_box("type", &curIndex, {"default", "optimized", "linear", "compute"}))
		{
			type = (curIndex == 0 ? DEF : (curIndex == 1 ? OPT : (curIndex == 2 ? LINEAR : COMP)));
			reset = true;
		}
	}

	if (drawer.header("Parameters"))
	{
		drawer.slider_float("sigma", &sigma, 0.01f, 5.0f);
	
		pushConstGraphics.gaussian_divisor = -0.5f / (sigma * sigma);
		pushConstCompute.gaussian_divisor  = pushConstGraphics.gaussian_divisor;	
	}
	
	if (drawer.header("Frametime"))
	{
		if (type == LINEAR || type == COMP)
		{
			drawer.text("first pass: %lf ms\n"
						"second pass: %lf ms\n"
						"total: %lf ms",
						frametime_first_pass,
						frametime_second_pass,
						frametime_first_pass + frametime_second_pass);
		}
		else
		{
			drawer.text("total: %lf ms", frametime);
		}
	}

	if (drawer.header("Average frametime"))
	{
		drawer.text("%llu frames", n_frames);
		if (type == LINEAR || type == COMP)
		{
			drawer.text("first pass: %lf ms\n"
						"second pass: %lf ms\n"
						"total: %lf ms",
						avg_frametime_first_pass,
						avg_frametime_second_pass,
						avg_frametime_first_pass + avg_frametime_second_pass);
		}
		else
		{
			drawer.text("total: %lf ms", avg_frametime);
		}
	}

	if (reset)
	{
		avg_frametime = 0.0;
		avg_frametime_first_pass = 0.0;
		avg_frametime_second_pass = 0.0;
		n_frames = 0;
	}
}

bool GaussianFilter::resize(uint32_t _width, uint32_t _height)
{
	if (!prepared)
	{
		return false;
	}

	get_render_context().handle_surface_changes();

	// Don't recreate the swapchain if the dimensions haven't changed
	if (width == get_render_context().get_surface_extent().width && height == get_render_context().get_surface_extent().height)
	{
		return false;
	}

	width  = get_render_context().get_surface_extent().width;
	height = get_render_context().get_surface_extent().height;

	pushConstGraphics.offset_width 	= 1.0f / width;
	pushConstGraphics.offset_height = 1.0f / height;
	pushConstCompute.width 			= width;
	pushConstCompute.height 		= height;

	prepared = false;

	// Ensure all operations on the device have been finished before destroying resources
	device->wait_idle();

	create_swapchain_buffers();
	setup_images();

	// Recreate the frame buffers
	vkDestroyImageView(device->get_handle(), depth_stencil.view, nullptr);
	vkDestroyImage(device->get_handle(), depth_stencil.image, nullptr);
	vkFreeMemory(device->get_handle(), depth_stencil.mem, nullptr);
	setup_depth_stencil();
	for (uint32_t i = 0; i < framebuffers.size(); i++)
	{
		vkDestroyFramebuffer(device->get_handle(), framebuffers[i], nullptr);
		vkDestroyFramebuffer(device->get_handle(), filter_pass_framebuffers[i], nullptr);
		framebuffers[i] = VK_NULL_HANDLE;
		filter_pass_framebuffers[i] = VK_NULL_HANDLE;
	}

	vkDestroyFramebuffer(device->get_handle(), main_pass.framebuffer, nullptr);
	main_pass.framebuffer = VK_NULL_HANDLE;

	vkDestroyFramebuffer(device->get_handle(), intermediate_filter_pass_framebuffer, nullptr);
	intermediate_filter_pass_framebuffer= VK_NULL_HANDLE;

	setup_framebuffer();

	if ((width > 0.0f) && (height > 0.0f))
	{
		if (gui)
		{
			gui->resize(width, height);
		}
	}

	avg_frametime = 0.0;
	avg_frametime_first_pass = 0.0;
	avg_frametime_second_pass = 0.0;
	n_frames = 0;

	rebuild_command_buffers();

	device->wait_idle();

	// Notify derived class
	view_changed();

	prepared = true;
	return true;
}

void GaussianFilter::setup_framebuffer()
{
	// present and filter final (or only) framebuffers
	{
		VkImageView attachment;

		VkFramebufferCreateInfo framebuffer_create_info = {};
		framebuffer_create_info.sType                   = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		framebuffer_create_info.pNext                   = NULL;
		framebuffer_create_info.attachmentCount         = 1;
		framebuffer_create_info.pAttachments            = &attachment;
		framebuffer_create_info.width                   = get_render_context().get_surface_extent().width;
		framebuffer_create_info.height                  = get_render_context().get_surface_extent().height;
		framebuffer_create_info.layers                  = 1;

		// Delete existing frame buffers
		if (framebuffers.size() > 0)
		{
			for (uint32_t i = 0; i < framebuffers.size(); i++)
			{
				if (framebuffers[i] != VK_NULL_HANDLE)
					vkDestroyFramebuffer(device->get_handle(), framebuffers[i], nullptr);
				if (filter_pass_framebuffers[i] != VK_NULL_HANDLE)
					vkDestroyFramebuffer(device->get_handle(), filter_pass_framebuffers[i], nullptr);
			}
		}

		// Create frame buffers for every swap chain image
		framebuffers.resize(render_context->get_render_frames().size());
		filter_pass_framebuffers.resize(framebuffers.size());
		for (uint32_t i = 0; i < framebuffers.size(); i++)
		{
			attachment = swapchain_buffers[i].view;
			framebuffer_create_info.renderPass = render_pass;
			VK_CHECK(vkCreateFramebuffer(device->get_handle(), &framebuffer_create_info, nullptr, &framebuffers[i]));
			framebuffer_create_info.renderPass = filter_pass;
			VK_CHECK(vkCreateFramebuffer(device->get_handle(), &framebuffer_create_info, nullptr, &filter_pass_framebuffers[i]));
		}
	}

	// main framebuffer
	{
		VkImageView attachment = main_pass.image_view->get_handle();

		VkFramebufferCreateInfo framebuffer_create_info = {};
		framebuffer_create_info.sType                   = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		framebuffer_create_info.pNext                   = nullptr;
		framebuffer_create_info.renderPass              = main_pass.render_pass;
		framebuffer_create_info.attachmentCount         = 1;
		framebuffer_create_info.pAttachments            = &attachment;
		framebuffer_create_info.width                   = get_render_context().get_surface_extent().width;
		framebuffer_create_info.height                  = get_render_context().get_surface_extent().height;
		framebuffer_create_info.layers                  = 1;

		if (main_pass.framebuffer != VK_NULL_HANDLE)
		{
			vkDestroyFramebuffer(device->get_handle(), main_pass.framebuffer, nullptr);
		}

		vkCreateFramebuffer(device->get_handle(), &framebuffer_create_info, nullptr, &main_pass.framebuffer);
	}

	// intermediate filter pass
	{
		VkImageView attachment = intermediate_image_view->get_handle();

		VkFramebufferCreateInfo framebuffer_create_info = {};
		framebuffer_create_info.sType                   = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		framebuffer_create_info.pNext                   = nullptr;
		framebuffer_create_info.renderPass              = intermediate_filter_pass;
		framebuffer_create_info.attachmentCount         = 1;
		framebuffer_create_info.pAttachments            = &attachment;
		framebuffer_create_info.width                   = get_render_context().get_surface_extent().width;
		framebuffer_create_info.height                  = get_render_context().get_surface_extent().height;
		framebuffer_create_info.layers                  = 1;

		if (intermediate_filter_pass_framebuffer != VK_NULL_HANDLE)
		{
			vkDestroyFramebuffer(device->get_handle(), intermediate_filter_pass_framebuffer, nullptr);
		}

		vkCreateFramebuffer(device->get_handle(), &framebuffer_create_info, nullptr, &intermediate_filter_pass_framebuffer);
	}
}

void GaussianFilter::setup_render_pass()
{	
	// present render pass (gui render pass)
	{
		VkAttachmentDescription attachment;

		// Color attachment
		attachment.flags		  = 0;
		attachment.format         = render_context->get_format();
		attachment.samples        = VK_SAMPLE_COUNT_1_BIT;
		attachment.loadOp         = VK_ATTACHMENT_LOAD_OP_LOAD;
		attachment.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
		attachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachment.initialLayout  = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		attachment.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentReference color_reference;
		color_reference.attachment	= 0;
		color_reference.layout		= VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass_description;
		subpass_description.flags 					= 0;
		subpass_description.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass_description.colorAttachmentCount    = 1;
		subpass_description.pColorAttachments       = &color_reference;
		subpass_description.pDepthStencilAttachment = nullptr;
		subpass_description.inputAttachmentCount    = 0;
		subpass_description.pInputAttachments       = nullptr;
		subpass_description.preserveAttachmentCount = 0;
		subpass_description.pPreserveAttachments    = nullptr;
		subpass_description.pResolveAttachments     = nullptr;

		VkSubpassDependency dependency;

		dependency.srcSubpass      = 0;
		dependency.dstSubpass      = VK_SUBPASS_EXTERNAL;
		dependency.srcStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.dstStageMask    = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		dependency.srcAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependency.dstAccessMask   = VK_ACCESS_MEMORY_READ_BIT;
		dependency.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		VkRenderPassCreateInfo render_pass_create_info = {};
		render_pass_create_info.sType                  = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		render_pass_create_info.attachmentCount        = 1;
		render_pass_create_info.pAttachments           = &attachment;
		render_pass_create_info.subpassCount           = 1;
		render_pass_create_info.pSubpasses             = &subpass_description;
		render_pass_create_info.dependencyCount        = 1;
		render_pass_create_info.pDependencies          = &dependency;

		VK_CHECK(vkCreateRenderPass(device->get_handle(), &render_pass_create_info, nullptr, &render_pass));
	}

	// filter pass
	{
		VkAttachmentDescription attachment;

		// Color attachment
		attachment.flags		  = 0;
		attachment.format         = render_context->get_format();
		attachment.samples        = VK_SAMPLE_COUNT_1_BIT;
		attachment.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachment.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
		attachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachment.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
		attachment.finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference color_reference;
		color_reference.attachment	= 0;
		color_reference.layout		= VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass_description;
		subpass_description.flags 					= 0;
		subpass_description.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass_description.colorAttachmentCount    = 1;
		subpass_description.pColorAttachments       = &color_reference;
		subpass_description.pDepthStencilAttachment = nullptr;
		subpass_description.inputAttachmentCount    = 0;
		subpass_description.pInputAttachments       = nullptr;
		subpass_description.preserveAttachmentCount = 0;
		subpass_description.pPreserveAttachments    = nullptr;
		subpass_description.pResolveAttachments     = nullptr;

		VkRenderPassCreateInfo render_pass_create_info = {};
		render_pass_create_info.sType                  = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		render_pass_create_info.attachmentCount        = 1;
		render_pass_create_info.pAttachments           = &attachment;
		render_pass_create_info.subpassCount           = 1;
		render_pass_create_info.pSubpasses             = &subpass_description;

		VK_CHECK(vkCreateRenderPass(device->get_handle(), &render_pass_create_info, nullptr, &filter_pass));
	}

	// intermediate filter pass and main render pass
	{
		VkAttachmentDescription attachment;

		// Color attachment
		attachment.flags		  = 0;
		attachment.format         = intermediate_image->get_format();
		attachment.samples        = VK_SAMPLE_COUNT_1_BIT;
		attachment.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachment.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
		attachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachment.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
		attachment.finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		VkAttachmentReference color_reference;
		color_reference.attachment	= 0;
		color_reference.layout		= VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass_description;
		subpass_description.flags 					= 0;
		subpass_description.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass_description.colorAttachmentCount    = 1;
		subpass_description.pColorAttachments       = &color_reference;
		subpass_description.pDepthStencilAttachment = nullptr;
		subpass_description.inputAttachmentCount    = 0;
		subpass_description.pInputAttachments       = nullptr;
		subpass_description.preserveAttachmentCount = 0;
		subpass_description.pPreserveAttachments    = nullptr;
		subpass_description.pResolveAttachments     = nullptr;

		VkSubpassDependency dependency;

		dependency.srcSubpass      = 0;
		dependency.dstSubpass      = VK_SUBPASS_EXTERNAL;
		dependency.srcStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.dstStageMask    = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		dependency.srcAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependency.dstAccessMask   = VK_ACCESS_SHADER_READ_BIT;
		dependency.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		VkRenderPassCreateInfo render_pass_create_info = {};
		render_pass_create_info.sType                  = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		render_pass_create_info.attachmentCount        = 1;
		render_pass_create_info.pAttachments           = &attachment;
		render_pass_create_info.subpassCount           = 1;
		render_pass_create_info.pSubpasses             = &subpass_description;
		render_pass_create_info.dependencyCount        = 1;
		render_pass_create_info.pDependencies          = &dependency;

		VK_CHECK(vkCreateRenderPass(device->get_handle(), &render_pass_create_info, nullptr, &intermediate_filter_pass));
		
		attachment.format = main_pass.image->get_format();
		VK_CHECK(vkCreateRenderPass(device->get_handle(), &render_pass_create_info, nullptr, &main_pass.render_pass));
	}
}

void GaussianFilter::prepare_pipelines()
{
	// Create pipeline layout.
	VkPushConstantRange range = vkb::initializers::push_constant_range(VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(pushConstGraphics), 0);

	VkPipelineLayoutCreateInfo layout_info = vkb::initializers::pipeline_layout_create_info(&descriptor_set_layouts.graphics_resolve);
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges = &range;
	VK_CHECK(vkCreatePipelineLayout(get_device().get_handle(), &layout_info, nullptr, &pipeline_layouts.graphics));

	VkPipelineVertexInputStateCreateInfo vertex_input;
	vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertex_input.pNext = nullptr;
	vertex_input.flags = 0;
	vertex_input.vertexBindingDescriptionCount = 0u;
	vertex_input.pVertexBindingDescriptions = nullptr;
	vertex_input.vertexAttributeDescriptionCount = 0u;
	vertex_input.pVertexAttributeDescriptions = nullptr;

	// Specify we will use triangle lists to draw geometry.
	VkPipelineInputAssemblyStateCreateInfo input_assembly;
	input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	input_assembly.pNext = nullptr;
	input_assembly.flags = 0;
	input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	input_assembly.primitiveRestartEnable = VK_FALSE;

	// Specify rasterization state.
	VkPipelineRasterizationStateCreateInfo raster;
	raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	raster.pNext = nullptr;
	raster.flags = 0;
	raster.depthClampEnable = VK_FALSE;
	raster.rasterizerDiscardEnable = VK_FALSE;
	raster.polygonMode = VK_POLYGON_MODE_FILL;
	raster.cullMode = 0;
	raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	raster.depthBiasEnable = VK_FALSE;
	raster.depthBiasConstantFactor = 0.0f;
	raster.depthBiasClamp = 0.0f;
	raster.depthBiasSlopeFactor = 0.0f;
	raster.lineWidth = 1.0f;

	VkPipelineMultisampleStateCreateInfo multisample_state;
	multisample_state.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisample_state.pNext = nullptr;
	multisample_state.flags = 0;
	multisample_state.rasterizationSamples 	= VK_SAMPLE_COUNT_1_BIT;
	multisample_state.sampleShadingEnable  	= VK_FALSE;
	multisample_state.minSampleShading		= 0.f;
	multisample_state.pSampleMask 			= nullptr;
	multisample_state.alphaToCoverageEnable	= VK_FALSE;
	multisample_state.alphaToOneEnable 		= VK_FALSE;

	VkPipelineDepthStencilStateCreateInfo depth_stencil;
	depth_stencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depth_stencil.pNext = nullptr;
	depth_stencil.flags = 0;
	depth_stencil.depthTestEnable = VK_FALSE;
	depth_stencil.depthWriteEnable = VK_FALSE;
	depth_stencil.depthCompareOp = VK_COMPARE_OP_ALWAYS;
	depth_stencil.depthBoundsTestEnable = VK_FALSE;
	depth_stencil.stencilTestEnable = VK_FALSE;
	depth_stencil.front = {VK_STENCIL_OP_ZERO, VK_STENCIL_OP_ZERO, VK_STENCIL_OP_ZERO, VK_COMPARE_OP_ALWAYS, 1, 1, 1};
	depth_stencil.back = depth_stencil.front;
	depth_stencil.minDepthBounds = 0.f;
	depth_stencil.maxDepthBounds = 1.f;

	// Our attachment will write to all color channels, but no blending is enabled.
	VkPipelineColorBlendAttachmentState blend_attachment = vkb::initializers::pipeline_color_blend_attachment_state(
		VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT, VK_FALSE);

	VkPipelineColorBlendStateCreateInfo blend;
	blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	blend.pNext = nullptr;
	blend.flags = 0;
	blend.logicOpEnable = VK_FALSE;
	blend.logicOp = VK_LOGIC_OP_NO_OP;
	blend.attachmentCount = 1;
	blend.pAttachments = &blend_attachment;

	// We will have one viewport and scissor box.
	VkPipelineViewportStateCreateInfo viewport = vkb::initializers::pipeline_viewport_state_create_info(1, 1);

	// Specify that these states will be dynamic, i.e. not part of pipeline state object.
	std::array<VkDynamicState, 2>    dynamics{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
	VkPipelineDynamicStateCreateInfo dynamic = vkb::initializers::pipeline_dynamic_state_create_info(dynamics.data(), vkb::to_u32(dynamics.size()));

	// Load our SPIR-V shaders.
	std::array<VkPipelineShaderStageCreateInfo, 2> shader_stages{};
	shader_stages[0] = load_shader(vertex_shader_path.data(), VK_SHADER_STAGE_VERTEX_BIT);

	// We need to specify the pipeline layout and the render pass description up front as well.
	VkGraphicsPipelineCreateInfo pipeline_create_info = vkb::initializers::pipeline_create_info();
	pipeline_create_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipeline_create_info.pNext = nullptr;
	pipeline_create_info.flags = 0;
	pipeline_create_info.pVertexInputState		= &vertex_input;
	pipeline_create_info.pInputAssemblyState	= &input_assembly;
	pipeline_create_info.pTessellationState		= nullptr;
	pipeline_create_info.pViewportState			= &viewport;
	pipeline_create_info.pRasterizationState	= &raster;
	pipeline_create_info.pMultisampleState		= &multisample_state;
	pipeline_create_info.pDepthStencilState		= &depth_stencil;
	pipeline_create_info.pColorBlendState		= &blend;
	pipeline_create_info.pDynamicState			= &dynamic;
	pipeline_create_info.layout = pipeline_layouts.graphics;
	pipeline_create_info.subpass = 0;
	pipeline_create_info.basePipelineHandle = VK_NULL_HANDLE;
	pipeline_create_info.basePipelineIndex = 0;

	pipeline_create_info.renderPass = filter_pass;
	// default shaders
	{	
		VkSpecializationMapEntry map_entry;
		map_entry.constantID = 0;
		map_entry.offset = 0;
		map_entry.size = sizeof(int32_t);

		int32_t data;

		VkSpecializationInfo spec_info;
		spec_info.mapEntryCount = 1;
		spec_info.pMapEntries = &map_entry;
		spec_info.dataSize = sizeof(data);
		spec_info.pData = &data;

		for (int i = 0; i < window_count; ++i)
		{
			data = i + 1;

			shader_stages[1] = load_shader(gaussian_filter_def_path.data(), VK_SHADER_STAGE_FRAGMENT_BIT);
			shader_stages[1].pSpecializationInfo = &spec_info;

			pipeline_create_info.stageCount = vkb::to_u32(shader_stages.size());
			pipeline_create_info.pStages 	= shader_stages.data();

			VK_CHECK(vkCreateGraphicsPipelines(get_device().get_handle(), VK_NULL_HANDLE, 1,
				&pipeline_create_info, nullptr, &gaussian_filter_def_pipelines[i]));
		}
	}

	// optimized shaders
	{	
		VkSpecializationMapEntry map_entry;
		map_entry.constantID = 0;
		map_entry.offset = 0;
		map_entry.size = sizeof(int32_t);

		int32_t data;

		VkSpecializationInfo spec_info;
		spec_info.mapEntryCount = 1;
		spec_info.pMapEntries = &map_entry;
		spec_info.dataSize = sizeof(data);
		spec_info.pData = &data;

		for (int i = 0; i < window_count; ++i)
		{
			data = i + 1;

			shader_stages[1] = load_shader(gaussian_filter_opt_path.data(), VK_SHADER_STAGE_FRAGMENT_BIT);
			shader_stages[1].pSpecializationInfo = &spec_info;

			pipeline_create_info.stageCount = vkb::to_u32(shader_stages.size());
			pipeline_create_info.pStages 	= shader_stages.data();

			VK_CHECK(vkCreateGraphicsPipelines(get_device().get_handle(), VK_NULL_HANDLE, 1,
				&pipeline_create_info, nullptr, &gaussian_filter_opt_pipelines[i]));
		}
	}

	// linear blur, vertical pass
	{	
		VkSpecializationMapEntry map_entry;
		map_entry.constantID = 0;
		map_entry.offset = 0;
		map_entry.size = sizeof(int32_t);

		int32_t data;

		VkSpecializationInfo spec_info;
		spec_info.mapEntryCount = 1;
		spec_info.pMapEntries = &map_entry;
		spec_info.dataSize = sizeof(data);
		spec_info.pData = &data;

		for (int i = 0; i < window_count; ++i)
		{
			data = i + 1;

			shader_stages[1] = load_shader(gaussian_filter_linear_vert_path.data(), VK_SHADER_STAGE_FRAGMENT_BIT);
			shader_stages[1].pSpecializationInfo = &spec_info;

			pipeline_create_info.stageCount = vkb::to_u32(shader_stages.size());
			pipeline_create_info.pStages 	= shader_stages.data();

			VK_CHECK(vkCreateGraphicsPipelines(get_device().get_handle(), VK_NULL_HANDLE, 1,
				&pipeline_create_info, nullptr, &gaussian_filter_linear_vert_pipelines[i]));
		}
	}

	pipeline_create_info.renderPass = intermediate_filter_pass;
	// linear blur, horizontal pass
	{	
		VkSpecializationMapEntry map_entry;
		map_entry.constantID = 0;
		map_entry.offset = 0;
		map_entry.size = sizeof(int32_t);

		int32_t data;

		VkSpecializationInfo spec_info;
		spec_info.mapEntryCount = 1;
		spec_info.pMapEntries = &map_entry;
		spec_info.dataSize = sizeof(data);
		spec_info.pData = &data;

		for (int i = 0; i < window_count; ++i)
		{
			data = i + 1;

			shader_stages[1] = load_shader(gaussian_filter_linear_horiz_path.data(), VK_SHADER_STAGE_FRAGMENT_BIT);
			shader_stages[1].pSpecializationInfo = &spec_info;

			pipeline_create_info.stageCount = vkb::to_u32(shader_stages.size());
			pipeline_create_info.pStages 	= shader_stages.data();

			VK_CHECK(vkCreateGraphicsPipelines(get_device().get_handle(), VK_NULL_HANDLE, 1,
				&pipeline_create_info, nullptr, &gaussian_filter_linear_horiz_pipelines[i]));
		}
	}

	// resolve graphics pipeline
	layout_info = vkb::initializers::pipeline_layout_create_info(&descriptor_set_layouts.graphics_resolve);
	layout_info.pushConstantRangeCount = 0;
	layout_info.pPushConstantRanges = nullptr;
	VK_CHECK(vkCreatePipelineLayout(get_device().get_handle(), &layout_info, nullptr, &pipeline_layouts.resolve));

	pipeline_create_info.layout = pipeline_layouts.resolve;
	pipeline_create_info.renderPass = filter_pass;
	shader_stages[1] = load_shader(resolve_fragment_shader_path.data(), VK_SHADER_STAGE_FRAGMENT_BIT);
	pipeline_create_info.stageCount = vkb::to_u32(shader_stages.size());
	pipeline_create_info.pStages = shader_stages.data();

	VK_CHECK(vkCreateGraphicsPipelines(get_device().get_handle(), VK_NULL_HANDLE, 1,
		&pipeline_create_info, nullptr, &resolve_pipeline));

	// main graphics pipeline
	pipeline_create_info.renderPass = main_pass.render_pass;

	VK_CHECK(vkCreateGraphicsPipelines(get_device().get_handle(), VK_NULL_HANDLE, 1,
		&pipeline_create_info, nullptr, &main_pass.pipeline));

	// compute pipelines
	range = vkb::initializers::push_constant_range(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(pushConstCompute), 0);

	layout_info = vkb::initializers::pipeline_layout_create_info(&descriptor_set_layouts.compute);
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges = &range;
	VK_CHECK(vkCreatePipelineLayout(get_device().get_handle(), &layout_info, nullptr, &pipeline_layouts.compute));

	VkComputePipelineCreateInfo compute_create_info = vkb::initializers::compute_pipeline_create_info(pipeline_layouts.compute);
	compute_create_info.basePipelineHandle = VK_NULL_HANDLE;
	compute_create_info.basePipelineIndex = 0;

	{
		std::array<VkSpecializationMapEntry, 4> map_entries;
		map_entries[0].constantID = 0;
		map_entries[0].offset = 0;
		map_entries[0].size = sizeof(int32_t);

		map_entries[1].constantID = 1;
		map_entries[1].offset = sizeof(int32_t);
		map_entries[1].size = sizeof(int32_t);

		map_entries[2].constantID = 2;
		map_entries[2].offset = 2 * sizeof(int32_t);
		map_entries[2].size = sizeof(int32_t);

		map_entries[3].constantID = 3;
		map_entries[3].offset = 3 * sizeof(int32_t);
		map_entries[3].size = sizeof(int32_t);

		std::array<int32_t, 4> data;
		
		data[0] = workgroup_axis_size;

		VkSpecializationInfo spec_info;
		spec_info.mapEntryCount = map_entries.size();
		spec_info.pMapEntries = map_entries.data();
		spec_info.dataSize = sizeof(data[0]) * data.size();
		spec_info.pData = data.data();

		for (int i = 0; i < window_count; ++i)
		{
			data[1] = i + 1;
			data[2] = workgroup_axis_size;
			data[3] = 1;

			compute_create_info.stage = load_shader(gaussian_filter_comp_path.data(), VK_SHADER_STAGE_COMPUTE_BIT);
			compute_create_info.stage.pSpecializationInfo = &spec_info;

			VK_CHECK(vkCreateComputePipelines(get_device().get_handle(), VK_NULL_HANDLE, 1, &compute_create_info, nullptr, &gaussian_filter_comp_first_pass_pipelines[i]));

			data[2] = 1;
			data[3] = workgroup_axis_size;

			VK_CHECK(vkCreateComputePipelines(get_device().get_handle(), VK_NULL_HANDLE, 1, &compute_create_info, nullptr, &gaussian_filter_comp_second_pass_pipelines[i]));
		}
	}
}

void GaussianFilter::setup_query_pool()
{
	VkQueryPoolCreateInfo query_pool_info;
	query_pool_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
	query_pool_info.pNext = nullptr;
	query_pool_info.flags = 0;
	query_pool_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
	query_pool_info.queryCount = 4;
	query_pool_info.pipelineStatistics = 0;
		
	VK_CHECK(vkCreateQueryPool(get_device().get_handle(), &query_pool_info, nullptr, &query_pool));
}

void GaussianFilter::setup_descriptor_set_layouts()
{
	// common set with sampler
	{
		VkDescriptorSetLayoutBinding sampler_binding = vkb::initializers::descriptor_set_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);
		VkDescriptorSetLayoutCreateInfo descriptor_set_layout_create_info = vkb::initializers::descriptor_set_layout_create_info(&sampler_binding, 1);
		VK_CHECK(vkCreateDescriptorSetLayout(get_device().get_handle(), &descriptor_set_layout_create_info, nullptr, &descriptor_set_layouts.graphics_resolve));
	}
	
	// compute-only set with storage_image
	{
		std::vector<VkDescriptorSetLayoutBinding> bindings = 
		{
			vkb::initializers::descriptor_set_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT, 0),
			vkb::initializers::descriptor_set_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1),
		};
		VkDescriptorSetLayoutCreateInfo descriptor_set_layout_create_info = vkb::initializers::descriptor_set_layout_create_info(bindings);
		VK_CHECK(vkCreateDescriptorSetLayout(get_device().get_handle(), &descriptor_set_layout_create_info, nullptr, &descriptor_set_layouts.compute));
	}
}

void GaussianFilter::setup_descriptor_pool()
{
	std::array<VkDescriptorPoolSize, 2> pool_size = 
	{
		vkb::initializers::descriptor_pool_size(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 6),
		vkb::initializers::descriptor_pool_size(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2),
	};

	VkDescriptorPoolCreateInfo descriptor_pool_create_info = vkb::initializers::descriptor_pool_create_info(pool_size.size(), pool_size.data(), 6);
	VK_CHECK(vkCreateDescriptorPool(get_device().get_handle(), &descriptor_pool_create_info, nullptr, &descriptor_pool));
}

void GaussianFilter::setup_descriptor_sets()
{
	// graphics descriptor sets
	VkDescriptorSetAllocateInfo allocate_info = vkb::initializers::descriptor_set_allocate_info(descriptor_pool, &descriptor_set_layouts.graphics_resolve, 1);
	VK_CHECK(vkAllocateDescriptorSets(get_device().get_handle(), &allocate_info, &descriptor_sets.graphics.first));
	VK_CHECK(vkAllocateDescriptorSets(get_device().get_handle(), &allocate_info, &descriptor_sets.graphics.second));
	
	// resolve descriptor set (same allocate_info)
	VK_CHECK(vkAllocateDescriptorSets(get_device().get_handle(), &allocate_info, &descriptor_sets.resolve));

	// main pass descriptor set (same allocate_info)
	VK_CHECK(vkAllocateDescriptorSets(get_device().get_handle(), &allocate_info, &main_pass.set));

	// compute descriptor sets
	allocate_info = vkb::initializers::descriptor_set_allocate_info(descriptor_pool, &descriptor_set_layouts.compute, 1);
	VK_CHECK(vkAllocateDescriptorSets(get_device().get_handle(), &allocate_info, &descriptor_sets.compute.first));
	VK_CHECK(vkAllocateDescriptorSets(get_device().get_handle(), &allocate_info, &descriptor_sets.compute.second));
}

void GaussianFilter::get_frame_time()
{
	uint64_t labels[4];
	uint32_t count = type == COMP || type == LINEAR ? 4 : 2;

	auto result = vkGetQueryPoolResults(get_device().get_handle(), query_pool, 0, count, sizeof(labels[0]) * count,
		&labels, sizeof(labels[0]), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
	
	if (type != COMP && type != LINEAR)
	{
		frametime = ((labels[1] & mask) - (labels[0] & mask)) * get_device().get_gpu().get_properties().limits.timestampPeriod * 1e-6;
		avg_frametime = (frametime + avg_frametime * n_frames) / (n_frames + 1);
	}
	else
	{
		frametime_first_pass = ((labels[1] & mask) - (labels[0] & mask)) * get_device().get_gpu().get_properties().limits.timestampPeriod * 1e-6;
		frametime_second_pass = ((labels[3] & mask) - (labels[2] & mask)) * get_device().get_gpu().get_properties().limits.timestampPeriod * 1e-6;
		avg_frametime_first_pass = (frametime_first_pass + avg_frametime_first_pass * n_frames) / (n_frames + 1);
		avg_frametime_second_pass = (frametime_second_pass + avg_frametime_second_pass * n_frames) / (n_frames + 1);
	}
	++n_frames;
}

void GaussianFilter::update_descriptor_sets()
{
	// main pass descriptor set
	{
		VkDescriptorImageInfo texture_descriptor;
		texture_descriptor.sampler = main_pass.texture.sampler;
		texture_descriptor.imageView = main_pass.texture.image->get_vk_image_view().get_handle();
		texture_descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		VkWriteDescriptorSet write_descriptor_set = vkb::initializers::write_descriptor_set(main_pass.set, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0, &texture_descriptor);
		vkUpdateDescriptorSets(get_device().get_handle(), 1, &write_descriptor_set, 0, VK_NULL_HANDLE);
	}

	// graphics first (and main) descriptor set
	{	
		VkDescriptorImageInfo texture_descriptor;
		texture_descriptor.sampler = main_pass.texture.sampler;
		texture_descriptor.imageView = main_pass.image_view->get_handle();
		texture_descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		VkWriteDescriptorSet write_descriptor_set = vkb::initializers::write_descriptor_set(descriptor_sets.graphics.first, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0, &texture_descriptor);
		vkUpdateDescriptorSets(get_device().get_handle(), 1, &write_descriptor_set, 0, VK_NULL_HANDLE);
	}

	// graphics second (additional, for linear blur) descriptor set
	{
		VkDescriptorImageInfo texture_descriptor;
		texture_descriptor.sampler = main_pass.texture.sampler;
		texture_descriptor.imageView = intermediate_image_view->get_handle();
		texture_descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		VkWriteDescriptorSet write_descriptor_set = vkb::initializers::write_descriptor_set(descriptor_sets.graphics.second, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0, &texture_descriptor);
		vkUpdateDescriptorSets(get_device().get_handle(), 1, &write_descriptor_set, 0, VK_NULL_HANDLE);
	}

	// resolve descriptor set (same allocate_info)
	{
		VkDescriptorImageInfo texture_descriptor;
		texture_descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		texture_descriptor.imageView = storage_output_image_view->get_handle();
		texture_descriptor.sampler = main_pass.texture.sampler;

		VkWriteDescriptorSet write_descriptor_set = vkb::initializers::write_descriptor_set(descriptor_sets.resolve, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0, &texture_descriptor);
		vkUpdateDescriptorSets(get_device().get_handle(), 1, &write_descriptor_set, 0, VK_NULL_HANDLE);
	}

	// compute descriptor sets
	{
		// first set
		{
			VkDescriptorImageInfo texture_descriptor;
			texture_descriptor.sampler = main_pass.texture.sampler;
			texture_descriptor.imageView = main_pass.image_view->get_handle();
			texture_descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

			VkWriteDescriptorSet write_descriptor_set = vkb::initializers::write_descriptor_set(descriptor_sets.compute.first, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0, &texture_descriptor);
			vkUpdateDescriptorSets(get_device().get_handle(), 1, &write_descriptor_set, 0, VK_NULL_HANDLE);

			texture_descriptor.sampler = VK_NULL_HANDLE;
			texture_descriptor.imageView = storage_intermediate_image_view->get_handle();
			texture_descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

			write_descriptor_set = vkb::initializers::write_descriptor_set(descriptor_sets.compute.first, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, &texture_descriptor);
			vkUpdateDescriptorSets(get_device().get_handle(), 1, &write_descriptor_set, 0, VK_NULL_HANDLE);
		}
		// second set
		{
			VkDescriptorImageInfo texture_descriptor;
			texture_descriptor.sampler = main_pass.texture.sampler;
			texture_descriptor.imageView = storage_intermediate_image_view->get_handle();
			texture_descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

			VkWriteDescriptorSet write_descriptor_set = vkb::initializers::write_descriptor_set(descriptor_sets.compute.second, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0, &texture_descriptor);
			vkUpdateDescriptorSets(get_device().get_handle(), 1, &write_descriptor_set, 0, VK_NULL_HANDLE);

			texture_descriptor.sampler = VK_NULL_HANDLE;
			texture_descriptor.imageView = storage_output_image_view->get_handle();
			texture_descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

			write_descriptor_set = vkb::initializers::write_descriptor_set(descriptor_sets.compute.second, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, &texture_descriptor);
			vkUpdateDescriptorSets(get_device().get_handle(), 1, &write_descriptor_set, 0, VK_NULL_HANDLE);
		}
	}
}

void GaussianFilter::setup_images()
{
	VkExtent3D extent = {get_render_context().get_surface_extent().width, 
		get_render_context().get_surface_extent().height, 1};

	main_pass.image = std::make_unique<vkb::core::Image>(get_device(), extent, VK_FORMAT_R8G8B8A8_UNORM,
		VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
		
	main_pass.image_view = std::make_unique<vkb::core::ImageView>(*main_pass.image,
		VK_IMAGE_VIEW_TYPE_2D, main_pass.image->get_format());

	intermediate_image = std::make_unique<vkb::core::Image>(get_device(), extent, VK_FORMAT_R8G8B8A8_UNORM,
		VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	
	intermediate_image_view = std::make_unique<vkb::core::ImageView>(*intermediate_image,
		VK_IMAGE_VIEW_TYPE_2D, intermediate_image->get_format());

	storage_intermediate_image = std::make_unique<vkb::core::Image>(get_device(), extent, VK_FORMAT_R8G8B8A8_UNORM,
		VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

	storage_intermediate_image_view = std::make_unique<vkb::core::ImageView>(*storage_intermediate_image,
		VK_IMAGE_VIEW_TYPE_2D, storage_intermediate_image->get_format());

	storage_output_image = std::make_unique<vkb::core::Image>(get_device(), extent, VK_FORMAT_R8G8B8A8_UNORM,
		VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

	storage_output_image_view = std::make_unique<vkb::core::ImageView>(*storage_output_image,
		VK_IMAGE_VIEW_TYPE_2D, storage_output_image->get_format());
}

std::unique_ptr<vkb::VulkanSample> create_gaussian_filter()
{
	return std::make_unique<GaussianFilter>();
}
