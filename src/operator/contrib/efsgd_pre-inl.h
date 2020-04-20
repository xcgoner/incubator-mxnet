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
 * \file efsgd_pre-inl.h
 * \brief Optimizer operators
 * \author 
 */
#ifndef MXNET_OPERATOR_CONTRIB_EFSGDPRE_INL_H_
#define MXNET_OPERATOR_CONTRIB_EFSGDPRE_INL_H_
#include <mxnet/operator.h>
#include <vector>
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {

struct EFSGDPreParam : public dmlc::Parameter<EFSGDPreParam> {
  float lr;
  float momentum;
  float wd;
  float rescale_grad;
  float clip_gradient;
  bool  nesterov;
  DMLC_DECLARE_PARAMETER(EFSGDPreParam) {
    DMLC_DECLARE_FIELD(lr)
    .describe("Learning rate");
    DMLC_DECLARE_FIELD(momentum)
    .set_range(0.0f, 1.0f)
    .set_default(0.0f)
    .describe("momentum");
    DMLC_DECLARE_FIELD(wd)
    .set_default(0.0f)
    .describe("Weight decay augments the objective function with a "
              "regularization term that penalizes large weights. "
              "The penalty scales with the square of the magnitude of each weight.");
    DMLC_DECLARE_FIELD(rescale_grad)
    .set_default(1.0f)
    .describe("rescale gradient as grad = rescale_grad*grad.");
    DMLC_DECLARE_FIELD(clip_gradient)
    .set_default(-1.0f)
    .describe("If greater than 0, clip gradient to "
              "grad = max(min(grad, -clip_gradient), clip_gradient). "
              "Otherwise turned off.");
    DMLC_DECLARE_FIELD(nesterov)
    .set_default(true)
    .describe("If true, use Nesterov momentum");
  }
};


// struct EFSGDPreUpdateKernel {
//   template<typename DType>
//   MSHADOW_XINLINE static void Map(index_t i,
//     DType* out_data, DType* e, DType* m, DType* m_wd,
//     const DType* weight_data, DType* grad_data,
//     const DType clip_gradient, const DType rescale_grad,
//     const DType momentum, const bool nesterov, 
//     const DType lr, const DType wd,  
//     const OpReqType req) {
//     using namespace mshadow_op;

//     grad_data[i] *= rescale_grad;

//     if (clip_gradient >= 0.0f) {
//       grad_data[i] = clip::Map(grad_data[i], clip_gradient);
//     }

//     grad_data[i] *= lr;

//     // momentum
//     m[i] = momentum * m[i] + grad_data[i];
//     if (nesterov) {
//       grad_data[i] += momentum * m[i];
//     }
//     else {
//       grad_data[i] = m[i];
//     }

//     // weight decay
//     m_wd[i] = momentum * m_wd[i] + lr * wd * weight_data[i];

//     DType weight = weight_data[i];

//     if (nesterov) {
//       weight *= (1.f - lr * wd);
//       weight -= momentum * m_wd[i];
//     }
//     else {
//       weight -= m_wd[i];
//     }

//     // error feedback
//     e[i] += grad_data[i];

//     KERNEL_ASSIGN(out_data[i], req, weight);
//   }
// };

// put lr outside momentum
struct EFSGDPreUpdateKernel {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i,
    DType* out_data, DType* e, DType* m, DType* m_wd,
    const DType* weight_data, DType* grad_data,
    const DType clip_gradient, const DType rescale_grad,
    const DType momentum, const bool nesterov, 
    const DType lr, const DType wd,  
    const OpReqType req) {
    using namespace mshadow_op;

    grad_data[i] *= rescale_grad;

    if (clip_gradient >= 0.0f) {
      grad_data[i] = clip::Map(grad_data[i], clip_gradient);
    }

    // momentum
    m[i] = momentum * m[i] + grad_data[i];
    if (nesterov) {
      grad_data[i] += momentum * m[i];
    }
    else {
      grad_data[i] = m[i];
    }

    // weight decay
    m_wd[i] = momentum * m_wd[i] + wd * weight_data[i];

    DType weight = weight_data[i];

    if (nesterov) {
      weight -= lr * (momentum * m_wd[i] + wd * weight_data[i]);
    }
    else {
      weight -= lr * m_wd[i];
    }

    // error feedback
    e[i] += lr * grad_data[i];

    KERNEL_ASSIGN(out_data[i], req, weight);
  }
};

template <typename xpu>
inline void EFSGDPreUpdate(const nnvm::NodeAttrs &attrs, const OpContext &ctx,
                          const std::vector<TBlob> &inputs,
                          const std::vector<OpReqType> &req,
                          const std::vector<TBlob> &outputs) {
  using namespace mxnet_op;
  const EFSGDPreParam &param = nnvm::get<EFSGDPreParam>(attrs.parsed);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    DType* weight_data = inputs[0].dptr<DType>();
    DType* grad_data = inputs[1].dptr<DType>();
    DType* e = inputs[2].dptr<DType>();
    DType* m = inputs[3].dptr<DType>();
    DType* m_wd = inputs[4].dptr<DType>();
    DType* out_data = outputs[0].dptr<DType>();

    Kernel<EFSGDPreUpdateKernel, xpu>::Launch(s, inputs[0].shape_.Size(),
      out_data, e, m, m_wd, weight_data, grad_data,
      static_cast<DType>(param.clip_gradient), static_cast<DType>(param.rescale_grad),
      static_cast<DType>(param.momentum), static_cast<bool>(param.nesterov),
      static_cast<DType>(param.lr), static_cast<DType>(param.wd), req[0]);
  });
}

// put lr outside momentum
struct MP_EFSGDPreUpdateKernel {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i,
    DType* out_data, float* e, float* m, float* m_wd,
    const DType* weight_data, DType* grad_data, float* weight32,
    const float clip_gradient, const float rescale_grad,
    const float momentum, const bool nesterov, 
    const float lr, const float wd,  
    const OpReqType req) {
    using namespace mshadow_op;

    float w = weight32[i];

    float g;
    if (clip_gradient >= 0.0f) {
      g = mshadow_op::clip::Map(rescale_grad
                    *static_cast<float>(grad_data[i]), clip_gradient);
    } else {
      g = rescale_grad*static_cast<float>(grad_data[i]);
    }

    // momentum
    m[i] = momentum * m[i] + g;
    if (nesterov) {
      g += momentum * m[i];
    }
    else {
      g = m[i];
    }

    // weight decay
    m_wd[i] = momentum * m_wd[i] + wd * w;

    if (nesterov) {
      w -= lr * (momentum * m_wd[i] + wd * w);
    }
    else {
      w -= lr * m_wd[i];
    }

    // error feedback
    e[i] += lr * grad_data[i];
    
    weight32[i] = w;
    KERNEL_ASSIGN(out_data[i], req, w);
  }
};

template<typename xpu>
inline void MP_EFSGDPreUpdate(const nnvm::NodeAttrs& attrs,
                         const OpContext &ctx,
                         const std::vector<TBlob> &inputs,
                         const std::vector<OpReqType> &req,
                         const std::vector<TBlob> &outputs) {
  using namespace mxnet_op;
  const EFSGDPreParam param = nnvm::get<EFSGDPreParam>(attrs.parsed);
  Stream<xpu>* s = ctx.get_stream<xpu>();
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    Tensor<xpu, 2, DType> weight = inputs[0].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> grad = inputs[1].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, float> e = inputs[2].FlatTo2D<xpu, float>(s);
    Tensor<xpu, 2, float> m = inputs[3].FlatTo2D<xpu, float>(s);
    Tensor<xpu, 2, float> m_wd = inputs[4].FlatTo2D<xpu, float>(s);
    Tensor<xpu, 2, float> weight32 = inputs[5].FlatTo2D<xpu, float>(s);
    Tensor<xpu, 2, DType> out = outputs[0].FlatTo2D<xpu, DType>(s);
    Kernel<MP_EFSGDPreUpdateKernel, xpu>::Launch(s, weight.shape_.Size(), out.dptr_,
      e.dptr_, m.dptr_, m_wd.dptr_, weight.dptr_, grad.dptr_, weight32.dptr_,
      param.clip_gradient, param.rescale_grad,
      param.momentum, param.nesterov,
      param.lr, param.wd, req[0]);
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_EFSGDPRE_INL_H_
