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

class TAAStats : public ApiVulkanSample
{
public:
	TAAStats();
	virtual ~TAAStats();

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
	static constexpr std::string_view taa_statistics_comp_path = 
		"taa_statistics/taa_comp.comp";
	static constexpr uint32_t workgroup_axis_size = 16u;
	std::array<VkPipeline, window_count> taa_statistics_comp_pipelines {};

	// common vertex shader for default, optimized and resolve shaders
	static constexpr std::string_view vertex_shader_path = "quad3_vert.vert";
	
	// default frag shaders and pipelines
	static constexpr std::string_view taa_statistics_def_path = "taa_statistics/taa_default.frag";
	std::array<VkPipeline, window_count> taa_statistics_def_pipelines {};
	
	// optimized frag shaders and pipelines
	static constexpr std::string_view taa_statistics_opt_path = "taa_statistics/taa_optimized.frag";
	std::array<VkPipeline, window_count> taa_statistics_opt_pipelines {};
	
	// resolve shader and pipeline
	static constexpr std::string_view resolve_fragment_shader_path = "simple.frag";
	VkPipeline resolve_pipeline {};

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
		VkDescriptorSet graphics;
		VkDescriptorSet compute;
		VkDescriptorSet resolve;
	} descriptor_sets;

	VkQueryPool query_pool;

	std::unique_ptr<vkb::core::Image> storage_image;
	std::unique_ptr<vkb::core::ImageView> storage_image_view;

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

	VkRenderPass filter_pass;
	std::vector<VkFramebuffer> filter_pass_framebuffers;

	enum Type 
	{
		DEF,
		OPT,
		COMP,
	} type = DEF;

	struct
	{
		float offset_width;
		float offset_height;
		float gamma;
	    float t;
	} pushConstGraphics;

	struct
	{
		uint32_t width;
		uint32_t height;
		float gamma;
		float t;
	} pushConstCompute;

	double frametime_filter  = 0.0;
	double avg_frametime_filter  = 0.0;

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

std::unique_ptr<vkb::VulkanSample> create_taa_stats();
