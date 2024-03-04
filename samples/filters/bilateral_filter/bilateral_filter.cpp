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

#include "bilateral_filter.h"
#include "core/command_buffer.h"

BilateralFilter::BilateralFilter()
{
	title = "Bilateral filters collection";
	// Because vulkan version on my mobile device is 1.1
	add_device_extension(VK_EXT_HOST_QUERY_RESET_EXTENSION_NAME);
}

BilateralFilter::~BilateralFilter()
{
	if (device)
	{
		for (int i = 0; i < bilateral_filter_def_pipelines.size(); ++i)
			vkDestroyPipeline(get_device().get_handle(), bilateral_filter_def_pipelines[i], nullptr);

		for (int i = 0; i < bilateral_filter_opt_pipelines.size(); ++i)
			vkDestroyPipeline(get_device().get_handle(), bilateral_filter_opt_pipelines[i], nullptr);

		for (int i = 0; i < bilateral_filter_comp_pipelines.size(); ++i)
			vkDestroyPipeline(get_device().get_handle(), bilateral_filter_comp_pipelines[i], nullptr);

		vkDestroyPipeline(get_device().get_handle(), resolve_pipeline, nullptr);

		vkDestroyPipelineLayout(get_device().get_handle(), pipeline_layouts.graphics, nullptr);
		vkDestroyPipelineLayout(get_device().get_handle(), pipeline_layouts.compute, nullptr);
		vkDestroyPipelineLayout(get_device().get_handle(), pipeline_layouts.resolve, nullptr);
		
		vkDestroyDescriptorSetLayout(get_device().get_handle(), descriptor_set_layouts.graphics_resolve, nullptr);
		vkDestroyDescriptorSetLayout(get_device().get_handle(), descriptor_set_layouts.compute, nullptr);

		main_texture.image.reset();

		vkDestroySampler(get_device().get_handle(), main_texture.sampler, nullptr);
		vkDestroySampler(get_device().get_handle(), nearest_sampler, nullptr);

		vkDestroyQueryPool(get_device().get_handle(), query_pool, nullptr);
	}
}

void BilateralFilter::request_gpu_features(vkb::PhysicalDevice &gpu)
{
	// We need to enable the command pool reset feature in the extension struct
	auto &requested_extension_features          = gpu.request_extension_features<VkPhysicalDeviceHostQueryResetFeaturesEXT>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES_EXT);
	requested_extension_features.hostQueryReset = VK_TRUE;
}

void BilateralFilter::prepare_pipelines()
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
	pipeline_create_info.renderPass = render_pass;
	pipeline_create_info.subpass = 0;
	pipeline_create_info.basePipelineHandle = VK_NULL_HANDLE;
	pipeline_create_info.basePipelineIndex = 0;

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

			shader_stages[1] = load_shader(bilateral_filter_def_path.data(), VK_SHADER_STAGE_FRAGMENT_BIT);
			shader_stages[1].pSpecializationInfo = &spec_info;

			pipeline_create_info.stageCount = vkb::to_u32(shader_stages.size());
			pipeline_create_info.pStages 	= shader_stages.data();

			VK_CHECK(vkCreateGraphicsPipelines(get_device().get_handle(), pipeline_cache, 1,
				&pipeline_create_info, nullptr, &bilateral_filter_def_pipelines[i]));
		}
	}

	// optimized shaders
	for (int i = 0; i < window_count; ++i)
	{
		shader_stages[1] = load_shader(fragment_shaders_optimized_path[i].data(), VK_SHADER_STAGE_FRAGMENT_BIT);

		pipeline_create_info.stageCount = vkb::to_u32(shader_stages.size());
		pipeline_create_info.pStages 	= shader_stages.data();

		VK_CHECK(vkCreateGraphicsPipelines(get_device().get_handle(), pipeline_cache, 1,
			&pipeline_create_info, nullptr, &bilateral_filter_opt_pipelines[i]));
	}

	// resolve graphics pipeline
	layout_info = vkb::initializers::pipeline_layout_create_info(&descriptor_set_layouts.graphics_resolve);
	layout_info.pushConstantRangeCount = 0;
	layout_info.pPushConstantRanges = nullptr;
	VK_CHECK(vkCreatePipelineLayout(get_device().get_handle(), &layout_info, nullptr, &pipeline_layouts.resolve));

	pipeline_create_info.layout = pipeline_layouts.resolve;
	shader_stages[1] = load_shader(resolve_fragment_shader_path.data(), VK_SHADER_STAGE_FRAGMENT_BIT);
	pipeline_create_info.stageCount = vkb::to_u32(shader_stages.size());
	pipeline_create_info.pStages = shader_stages.data();

	VK_CHECK(vkCreateGraphicsPipelines(get_device().get_handle(), pipeline_cache, 1,
		&pipeline_create_info, nullptr, &resolve_pipeline));

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
		std::array<VkSpecializationMapEntry, 2> map_entries;
		map_entries[0].constantID = 0;
		map_entries[0].offset = 0;
		map_entries[0].size = sizeof(int32_t);

		map_entries[1].constantID = 1;
		map_entries[1].offset = sizeof(int32_t);
		map_entries[1].size = sizeof(int32_t);

		std::array<int32_t, 2> data;
		data[0] = workgroup_axis_size;

		VkSpecializationInfo spec_info;
		spec_info.mapEntryCount = map_entries.size();
		spec_info.pMapEntries = map_entries.data();
		spec_info.dataSize = sizeof(data[0]) * data.size();
		spec_info.pData = data.data();

		for (int i = 0; i < window_count; ++i)
		{
			data[1] = i + 1;

			compute_create_info.stage = load_shader(bilateral_filter_comp_path.data(), VK_SHADER_STAGE_COMPUTE_BIT);
			compute_create_info.stage.pSpecializationInfo = &spec_info;

			VK_CHECK(vkCreateComputePipelines(get_device().get_handle(), pipeline_cache, 1, &compute_create_info, nullptr, &bilateral_filter_comp_pipelines[i]));
		}
	}
}

bool BilateralFilter::prepare(const vkb::ApplicationOptions &options)
{
	if (!ApiVulkanSample::prepare(options))
	{
		return false;
	}

	main_texture = load_texture("textures/lavaplanet_color_rgba.ktx", vkb::sg::Image::Color);

	VkSamplerCreateInfo sampler_info;
	sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	sampler_info.pNext = nullptr;
	sampler_info.flags = 0;
	sampler_info.magFilter = VK_FILTER_NEAREST;
	sampler_info.minFilter = VK_FILTER_NEAREST;
	sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
	sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
	sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
	sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
	sampler_info.mipLodBias = 0.0f;
	sampler_info.anisotropyEnable = VK_FALSE;
	sampler_info.maxAnisotropy = 1.0f;
	sampler_info.compareEnable = VK_FALSE;
	sampler_info.compareOp = VK_COMPARE_OP_ALWAYS;
	sampler_info.minLod = 0.0f;
	sampler_info.maxLod = VK_LOD_CLAMP_NONE;
	sampler_info.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
	sampler_info.unnormalizedCoordinates = VK_FALSE;

	vkCreateSampler(get_device().get_handle(), &sampler_info, nullptr, &nearest_sampler);

	current_sampler = nearest_sampler;

	storage_image = std::make_unique<vkb::core::Image>(get_device(), VkExtent3D{width, height, 1}, 
		VK_FORMAT_R8G8B8A8_UNORM, 
		VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	storage_image_view = std::make_unique<vkb::core::ImageView>(*storage_image,
		VK_IMAGE_VIEW_TYPE_2D, storage_image->get_format());

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

void BilateralFilter::on_update_ui_overlay(vkb::Drawer &drawer)
{
	if (drawer.header("Select shader"))
	{
		uint32_t prev_pipeline_id = pipeline_id;
		if (drawer.button("3x3"))
		{
			std::cout << "3x3" << std::endl;
			pipeline_id = 0;
		}
		ImGui::SameLine();
		if (drawer.button("5x5"))
		{
			std::cout << "5x5" << std::endl;
			pipeline_id = 1;
		}
		ImGui::SameLine();
		if (drawer.button("7x7"))
		{
			std::cout << "7x7" << std::endl;
			pipeline_id = 2;
		}

		int32_t curIndex = type == COMP ? 2 : type == OPT;
		if (drawer.combo_box("type", &curIndex, {"default", "optimized", "compute"}))
		{
			std::cout << "type" << std::endl;
			type = curIndex == 0 ? DEF : curIndex == 1 ? OPT : COMP;
		}

		curIndex = current_sampler == nearest_sampler;
		if (drawer.combo_box("sampler", &curIndex, {"linear", "nearest"}))
		{
			std::cout << "sampler" << std::endl;
			current_sampler = curIndex ? nearest_sampler : main_texture.sampler;
		}

		if (drawer.slider_int("draw calls count", &draw_count, 1, 256))
		{
			std::cout << "draw calls count" << std::endl;
		}
	}

	if (drawer.header("Parameters"))
	{
		drawer.slider_float("sigma_d", &sigma_d, 0.01f, 5.0f);
		drawer.slider_float("sigma_r", &sigma_r, 0.01f, 1.0f);

		pushConstGraphics.gaussian_divisor    	= -0.5f / (sigma_d * sigma_d);
		pushConstGraphics.intensities_divisor 	= -0.5f / (sigma_r * sigma_r);

		pushConstCompute.gaussian_divisor		= pushConstGraphics.gaussian_divisor;
		pushConstCompute.intensities_divisor	= pushConstGraphics.intensities_divisor;
	}
	
	if (drawer.header("Frametime"))
	{
		drawer.text("%lf ms", frametime);
	}
}

bool BilateralFilter::resize(uint32_t width, uint32_t height)
{
	pushConstGraphics.offset_width 	= 1.0f / width;
	pushConstGraphics.offset_height = 1.0f / height;
	pushConstCompute.width 			= width;
	pushConstCompute.height 		= height;

	storage_image.reset(new vkb::core::Image(get_device(), VkExtent3D{width, height, 1}, 
		VK_FORMAT_R8G8B8A8_UNORM,
		VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VMA_MEMORY_USAGE_GPU_ONLY));

	storage_image_view.reset(new vkb::core::ImageView(*storage_image, 
		VK_IMAGE_VIEW_TYPE_2D, storage_image->get_format()));
	
	ApiVulkanSample::resize(width, height);
	
	return true;
}

void BilateralFilter::setup_framebuffer()
{
	VkImageView attachment;

	VkFramebufferCreateInfo framebuffer_create_info = {};
	framebuffer_create_info.sType                   = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
	framebuffer_create_info.pNext                   = NULL;
	framebuffer_create_info.renderPass              = render_pass;
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
			{
				vkDestroyFramebuffer(device->get_handle(), framebuffers[i], nullptr);
			}
		}
	}

	// Create frame buffers for every swap chain image
	framebuffers.resize(render_context->get_render_frames().size());
	for (uint32_t i = 0; i < framebuffers.size(); i++)
	{
		attachment = swapchain_buffers[i].view;
		VK_CHECK(vkCreateFramebuffer(device->get_handle(), &framebuffer_create_info, nullptr, &framebuffers[i]));
	}
}

void BilateralFilter::setup_query_pool()
{
	VkQueryPoolCreateInfo query_pool_info;
	query_pool_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
	query_pool_info.pNext = nullptr;
	query_pool_info.flags = 0;
	query_pool_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
	query_pool_info.queryCount = 2;
	query_pool_info.pipelineStatistics = 0;
		
	VK_CHECK(vkCreateQueryPool(get_device().get_handle(), &query_pool_info, nullptr, &query_pool));
}

void BilateralFilter::setup_descriptor_set_layouts()
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

void BilateralFilter::setup_descriptor_pool()
{
	std::array<VkDescriptorPoolSize, 2> pool_size = 
	{
		vkb::initializers::descriptor_pool_size(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3),
		vkb::initializers::descriptor_pool_size(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1),
	};

	VkDescriptorPoolCreateInfo descriptor_pool_create_info = vkb::initializers::descriptor_pool_create_info(pool_size.size(), pool_size.data(), 3);
	VK_CHECK(vkCreateDescriptorPool(get_device().get_handle(), &descriptor_pool_create_info, nullptr, &descriptor_pool));
}

void BilateralFilter::setup_descriptor_sets()
{
	// graphics descriptor set
	VkDescriptorSetAllocateInfo allocate_info = vkb::initializers::descriptor_set_allocate_info(descriptor_pool, &descriptor_set_layouts.graphics_resolve, 1);
	VK_CHECK(vkAllocateDescriptorSets(get_device().get_handle(), &allocate_info, &descriptor_sets.graphics));
	
	// resolve descriptor set (same allocate_info)
	VK_CHECK(vkAllocateDescriptorSets(get_device().get_handle(), &allocate_info, &descriptor_sets.resolve));
	
	// compute descriptor set
	allocate_info = vkb::initializers::descriptor_set_allocate_info(descriptor_pool, &descriptor_set_layouts.compute, 1);
	VK_CHECK(vkAllocateDescriptorSets(get_device().get_handle(), &allocate_info, &descriptor_sets.compute));
}

void BilateralFilter::get_frame_time()
{
	uint64_t labels[2];

	uint32_t validBits = get_device().get_gpu().get_queue_family_properties()[get_device().get_queue_family_index(VK_QUEUE_GRAPHICS_BIT)].timestampValidBits;
	LOGI(validBits);
	assert(validBits);
	uint64_t mask = ~0ULL >> 64u - validBits;

	auto result = vkGetQueryPoolResults(get_device().get_handle(), query_pool, 0, 2, sizeof(labels[0]) * 2,
		&labels, sizeof(labels[0]), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
	
	frametime = ((labels[1] & mask) - (labels[0] & mask)) * get_device().get_gpu().get_properties().limits.timestampPeriod * 1e-6;
}

void BilateralFilter::update_descriptor_sets()
{
	assert(current_sampler != VK_NULL_HANDLE);

	// graphics descriptor set
	{	
		VkDescriptorImageInfo texture_descriptor = create_descriptor(main_texture);
		texture_descriptor.sampler = current_sampler;

		VkWriteDescriptorSet write_descriptor_set = vkb::initializers::write_descriptor_set(descriptor_sets.graphics, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0, &texture_descriptor);
		vkUpdateDescriptorSets(get_device().get_handle(), 1, &write_descriptor_set, 0, VK_NULL_HANDLE);
	}

	// resolve descriptor set (same allocate_info)
	{
		VkDescriptorImageInfo texture_descriptor;
		texture_descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		texture_descriptor.imageView = storage_image_view->get_handle();
		texture_descriptor.sampler = current_sampler;

		VkWriteDescriptorSet write_descriptor_set = vkb::initializers::write_descriptor_set(descriptor_sets.resolve, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0, &texture_descriptor);
		vkUpdateDescriptorSets(get_device().get_handle(), 1, &write_descriptor_set, 0, VK_NULL_HANDLE);
	}

	// compute descriptor set
	{
		VkDescriptorImageInfo texture_descriptor = create_descriptor(main_texture);
		texture_descriptor.sampler = current_sampler;

		VkWriteDescriptorSet write_descriptor_set = vkb::initializers::write_descriptor_set(descriptor_sets.compute, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0, &texture_descriptor);
		vkUpdateDescriptorSets(get_device().get_handle(), 1, &write_descriptor_set, 0, VK_NULL_HANDLE);

		texture_descriptor.sampler = VK_NULL_HANDLE;
		texture_descriptor.imageView = storage_image_view->get_handle();
		texture_descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

		write_descriptor_set = vkb::initializers::write_descriptor_set(descriptor_sets.compute, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, &texture_descriptor);
		vkUpdateDescriptorSets(get_device().get_handle(), 1, &write_descriptor_set, 0, VK_NULL_HANDLE);
	}
}

void BilateralFilter::setup_render_pass()
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
	dependency.srcStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
	dependency.dstStageMask    = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
	dependency.srcAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
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

void BilateralFilter::build_command_buffers()
{
	update_descriptor_sets();
	
	VkCommandBufferBeginInfo command_buffer_begin_info = vkb::initializers::command_buffer_begin_info();

	// Clear color and depth values.
	VkClearValue clear_values[2];
	clear_values[0].color        = {{0.0f, 0.0f, 0.0f, 0.0f}};
	clear_values[1].depthStencil = {0.0f, 0};

	// Begin the render pass.
	VkRenderPassBeginInfo render_pass_begin_info    = vkb::initializers::render_pass_begin_info();
	render_pass_begin_info.renderPass               = render_pass;
	render_pass_begin_info.renderArea.offset.x      = 0;
	render_pass_begin_info.renderArea.offset.y      = 0;
	render_pass_begin_info.renderArea.extent.width  = width;
	render_pass_begin_info.renderArea.extent.height = height;
	render_pass_begin_info.clearValueCount          = 2;
	render_pass_begin_info.pClearValues             = clear_values;

	for (int32_t i = 0; i < draw_cmd_buffers.size(); ++i)
	{
		auto cmd = draw_cmd_buffers[i];

		// Begin command buffer.
		vkBeginCommandBuffer(cmd, &command_buffer_begin_info);

		vkCmdResetQueryPool(cmd, query_pool, 0, 2);

		if (type == COMP)
		{
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bilateral_filter_comp_pipelines[pipeline_id]);

			VkImageMemoryBarrier image_barrier;
			image_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			image_barrier.pNext = nullptr;
			image_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			image_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			image_barrier.srcAccessMask = VK_ACCESS_NONE;
			image_barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			image_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			image_barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
			image_barrier.image = storage_image->get_handle();
			image_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
			
			vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &image_barrier);

			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layouts.compute, 0, 1, &descriptor_sets.compute, 0, nullptr);
			
			vkCmdPushConstants(cmd, pipeline_layouts.compute, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstCompute), &pushConstCompute);

			uint32_t x_size = width / workgroup_axis_size + (width % workgroup_axis_size != 0);
			uint32_t y_size = height / workgroup_axis_size + (height % workgroup_axis_size != 0);

			vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool, 0);
			vkCmdDispatch(cmd, x_size, y_size, 1);
			vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool, 1);

			image_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			image_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			image_barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
			image_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

			vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 
				VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &image_barrier);
		}

		// Set framebuffer for this command buffer.
		render_pass_begin_info.framebuffer = framebuffers[i];

		// We will add draw commands in the same command buffer.
		vkCmdBeginRenderPass(cmd, &render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);

		// Set viewport dynamically
		VkViewport viewport = vkb::initializers::viewport(static_cast<float>(width), static_cast<float>(height), 0.0f, 1.0f);
		vkCmdSetViewport(cmd, 0, 1, &viewport);

		// Set scissor dynamically
		VkRect2D scissor = vkb::initializers::rect2D(width, height, 0, 0);
		vkCmdSetScissor(cmd, 0, 1, &scissor);

		switch (type)
		{
		case DEF:
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, bilateral_filter_def_pipelines[pipeline_id]);

			vkCmdPushConstants(cmd, pipeline_layouts.graphics, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pushConstGraphics), &pushConstGraphics);

			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layouts.graphics, 0, 1, &descriptor_sets.graphics, 0, nullptr);
			
			vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool, 0);

			vkCmdDraw(cmd, 3, draw_count, 0, 0);

			vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool, 1);
			break;
		case OPT:
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, bilateral_filter_opt_pipelines[pipeline_id]);

			vkCmdPushConstants(cmd, pipeline_layouts.graphics, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pushConstGraphics), &pushConstGraphics);

			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layouts.graphics, 0, 1, &descriptor_sets.graphics, 0, nullptr);
			
			vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool, 0);

			vkCmdDraw(cmd, 3, draw_count, 0, 0);

			vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool, 1);
			break;
		case COMP:
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, resolve_pipeline);

			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layouts.resolve, 0, 1, &descriptor_sets.resolve, 0, nullptr);
			
			vkCmdDraw(cmd, 3, 1, 0, 0);
			break;
		default:
			LOGE("Unknown type");
			exit(1);
		}
				
		// Draw user interface.
		draw_ui(cmd);

		// Complete render pass.
		vkCmdEndRenderPass(cmd);

		// Complete the command buffer.
		VK_CHECK(vkEndCommandBuffer(cmd));
	}
}

void BilateralFilter::render(float delta_time)
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

std::unique_ptr<vkb::VulkanSample> create_bilateral_filter()
{
	return std::make_unique<BilateralFilter>();
}
