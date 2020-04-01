/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2018 by Contributors
 * \file efsgd_pre.cc
 * \brief Optimizer operators
 * \author 
 */
#include "./efsgd_pre-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(EFSGDPreParam);

NNVM_REGISTER_OP(efsgd_pre_update)
.describe(R"code(Update function for EFSGD optimizer. 
)code" ADD_FILELINE)
.set_num_inputs(5)
.set_num_outputs(1)
.set_attr_parser(ParamParser<EFSGDPreParam>)
.set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<5, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<5, 1>)
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    return std::vector<uint32_t>{1,2,3,4};
  })
.set_attr<FCompute>("FCompute<cpu>", EFSGDPreUpdate<cpu>)
.add_argument("weight", "NDArray-or-Symbol", "Weight")
.add_argument("grad", "NDArray-or-Symbol", "Gradient")
.add_argument("e", "NDArray-or-Symbol", "Remaining error")
.add_argument("m", "NDArray-or-Symbol", "Momentum")
.add_argument("m_wd", "NDArray-or-Symbol", "Momentum of weight decay")
.add_arguments(EFSGDPreParam::__FIELDS__());


}  // namespace op
}  // namespace mxnet
