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

#pragma once

#include "api_vulkan_sample.h"

#include <utility>

class GaussianFilter : public ApiVulkanSample
{
  public:
	GaussianFilter();
	virtual ~GaussianFilter();

	// Override basic framework functionality
	virtual void build_command_buffers() override;
	virtual void render(float delta_time) override;
	virtual bool prepare(const vkb::ApplicationOptions &options) override;
	virtual void on_update_ui_overlay(vkb::Drawer &drawer) override;
	virtual bool resize(uint32_t width, uint32_t height) override;
	virtual void setup_framebuffer() override;
	virtual void setup_render_pass() override;
private:
	static constexpr std::string_view texture_path = "textures/Lenna.ktx";

	// Sample specific data
	uint32_t pipeline_id = 0; // same for all shader kinds

	// how many shaders of each kind
	static constexpr uint32_t window_count = 3;

	// compute shaders and pipelines
	static constexpr std::string_view gaussian_filter_comp_path = "gaussian_filter/gaussian_blur_comp.comp";
	static constexpr uint32_t workgroup_axis_size = 128u;
	std::array<VkPipeline, window_count> gaussian_filter_comp_first_pass_pipelines {};
    std::array<VkPipeline, window_count> gaussian_filter_comp_second_pass_pipelines {};

	// common vertex shader for default, optimized and resolve shaders
	static constexpr std::string_view vertex_shader_path = "quad3_vert.vert";
	
	// default frag shaders and pipelines
	static constexpr std::string_view gaussian_filter_def_path = "gaussian_filter/gaussian_blur.frag";
	std::array<VkPipeline, window_count> gaussian_filter_def_pipelines {};
	
	// optimized frag shaders and pipelines
	static constexpr std::string_view gaussian_filter_opt_path = "gaussian_filter/gaussian_blur_optimized.frag";
	std::array<VkPipeline, window_count> gaussian_filter_opt_pipelines {};

	// vertical linear filter shaders and pipelines
	static constexpr std::string_view gaussian_filter_linear_vert_path = "gaussian_filter/gaussian_blur_linear_vert.frag";
	std::array<VkPipeline, window_count> gaussian_filter_linear_vert_pipelines {};

	// vertical linear filter shaders and pipelines
	static constexpr std::string_view gaussian_filter_linear_horiz_path = "gaussian_filter/gaussian_blur_linear_horiz.frag";
	std::array<VkPipeline, window_count> gaussian_filter_linear_horiz_pipelines {};

	// resolve shader and pipeline
	static constexpr std::string_view resolve_fragment_shader_path = "simple.frag";
	VkPipeline 				resolve_pipeline {};

	struct
	{
		VkPipelineLayout resolve;
		VkPipelineLayout graphics;
		VkPipelineLayout compute;
	} pipeline_layouts;

	struct 
	{
		VkDescriptorSetLayout graphics_resolve; // common layout
		VkDescriptorSetLayout compute;
	} descriptor_set_layouts;

	struct
	{
		std::pair<VkDescriptorSet, VkDescriptorSet> graphics;
		std::pair<VkDescriptorSet, VkDescriptorSet> compute;
		VkDescriptorSet resolve;
	} descriptor_sets;

	VkQueryPool query_pool;

	std::unique_ptr<vkb::core::Image> storage_intermediate_image;
	std::unique_ptr<vkb::core::ImageView> storage_intermediate_image_view;

    std::unique_ptr<vkb::core::Image> storage_output_image;
	std::unique_ptr<vkb::core::ImageView> storage_output_image_view;

	struct
	{
		Texture 								texture;
		std::unique_ptr<vkb::core::Image> 		image;
		std::unique_ptr<vkb::core::ImageView> 	image_view;
		VkFramebuffer							framebuffer;
		VkRenderPass 							render_pass;
		VkDescriptorSet							set;
		VkPipeline								pipeline;
	} main_pass {};

	std::unique_ptr<vkb::core::Image> 		intermediate_image;
	std::unique_ptr<vkb::core::ImageView> 	intermediate_image_view;

	VkRenderPass filter_pass;
	VkRenderPass intermediate_filter_pass;
	
	VkFramebuffer intermediate_filter_pass_framebuffer = VK_NULL_HANDLE;
	std::vector<VkFramebuffer> filter_pass_framebuffers;

	enum Type 
	{
		DEF,
		OPT,
		COMP,
		LINEAR,
	} type = DEF;

	struct
	{
		float offset_width;
		float offset_height;
		float gaussian_divisor;
	} pushConstGraphics;

	struct
	{
		uint32_t width;
		uint32_t height;
		float gaussian_divisor;
	} pushConstCompute;

	float sigma = 3.0f;

	double frametime = 0.0;
	double frametime_first_pass = 0.0;
	double frametime_second_pass = 0.0;

	double avg_frametime = 0.0;
	double avg_frametime_first_pass = 0.0;
	double avg_frametime_second_pass = 0.0;

	uint64_t n_frames = 0;

	uint64_t mask; 

	void prepare_pipelines();
	void setup_query_pool();
	void setup_descriptor_set_layouts();
	void setup_descriptor_pool();
	void setup_descriptor_sets();
	void get_frame_time();
	void update_descriptor_sets();
	void setup_images();
};

std::unique_ptr<vkb::VulkanSample> create_gaussian_filter();
