// Copyright 2026 KIT-ISAS
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <gtest/gtest.h>

#include <string>

#include "ament_index_cpp/get_resource.hpp"

// Smoke test: the libsurvive component must be registered with the ament index
// so that rclcpp_components can discover and load it at runtime.
TEST(HolodeckPkg, component_is_registered)
{
  std::string content;
  std::string prefix;
  const bool found = ament_index_cpp::get_resource(
    "rclcpp_components", "holodeck_pkg", content, &prefix);
  ASSERT_TRUE(found) << "holodeck_pkg has no rclcpp_components resource registered";
  EXPECT_NE(content.find("libsurvive_ros2::Component"), std::string::npos)
    << "component class not listed in the rclcpp_components resource: " << content;
}
