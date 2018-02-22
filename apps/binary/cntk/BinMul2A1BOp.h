//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "BG2A1B.h"
#include "CNTKLibrary.h"
#include <stdio.h>

using namespace CNTK;

void binarize_array_T(const float *input, int x, int y, int64_t *binary)
{
    for (int i = 0; i < x; ++i) {
        for (int j = 0; j < y; ++j) {
            int index = y*i + j;
            int block = index/64;
            int bit = index%64;
            float input_val = input[j*x + i]; 
            if (input_val > 0) {
                binary[block] |= ((uint64_t) 1 << bit);
            } else {
                binary[block] &= ~((uint64_t) 1 << bit);
            }
        }
    }   
}

class BinMul2A1B final : public Function
{
public:
    static FunctionPtr Create(const Variable& leftOperand, const Variable& rightOperand, const Dictionary& attributes, const std::wstring& name)
    {
        return AsComposite(MakeSharedObject<BinMul2A1B>(leftOperand, rightOperand, attributes, name));
    }

    static FunctionPtr Create(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name)
    {
        return Create(leftOperand, rightOperand, Dictionary(), name);
    }

    BinMul2A1B(const Variable& leftOperand, const Variable& rightOperand, const Dictionary& attributes, const std::wstring& name)
        : Function({ leftOperand, rightOperand }, Dictionary(attributes), name), Attr(Dictionary(attributes))
    {
        m = Attr[m_key].Value<int>();
        n = Attr[n_key].Value<int>();
        k = Attr[k_key].Value<int>();
        const NDArrayViewPtr& B_array = (leftOperand.Value());
        B_data = B_array->DataBuffer<float>();
        assert(n % 64 == 0);
        binary_B = (int64_t *) malloc((n/64)*k*sizeof(int64_t));
        binarize_array_T(B_data, n, k, binary_B);
        Executor = new BinGemm2A1B(binary_B, m, n, k);
    }

private:


    void MatrixMultiply(const NDArrayViewPtr& leftMatrix, const NDArrayViewPtr& rightMatrix, NDArrayViewPtr& outputMatrix, bool transposeRight = false)
    {
        auto leftBuffer = leftMatrix->DataBuffer<float>();
        auto rightBuffer = rightMatrix->DataBuffer<float>();
        auto outBuffer = outputMatrix->WritableDataBuffer<float>();
        Executor->realize(rightBuffer, outBuffer);
    }

    BackPropStatePtr Forward(const std::vector<ValuePtr>& inputValues,
                             std::unordered_map<Variable, ValuePtr>& outputs,
                             const DeviceDescriptor& computeDevice,
                             const std::unordered_set<Variable>& /*outputsToRetainBackwardStateFor*/) override
    {
        auto leftOperandData = inputValues[0]->Data();
        auto rightOperandData = inputValues[1]->Data();

        // Allocate outputValue if needed
        auto& outputValue = outputs[this->Output()];
        if (outputValue == nullptr)
        {
            auto numOutRows = leftOperandData->Shape()[0];
            auto numOutCols = rightOperandData->Shape()[rightOperandData->Shape().Rank() - 1];
            outputValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(DataType::Float, NDShape({ numOutRows , numOutCols }), computeDevice));
        }

        auto outputData = outputValue->Data();
        MatrixMultiply(leftOperandData, rightOperandData, outputData);

        // Let's save the right input's Value in the BackPropSate to be used in the backward pass for computing gradients
        return MakeSharedObject<BackPropState>(this->shared_from_this(), computeDevice, std::unordered_map<Variable, ValuePtr>({ {Inputs()[1], inputValues[1] } }));
    }

    void Backward(const BackPropStatePtr& state,
                  const std::unordered_map<Variable, ValuePtr>& rootGradientValues,
                  std::unordered_map<Variable, ValuePtr>& backPropagatedGradientValuesForInputs) override
    {
        auto leftInputVariable = Inputs()[0];
        auto rightInputVariable = Inputs()[1];
        if (backPropagatedGradientValuesForInputs.find(rightInputVariable) != backPropagatedGradientValuesForInputs.end())
            std::runtime_error("BinMul2A1B does not support computing gradient wrt right operand");

        auto rightInputData = state->SavedForwardPropValues().at(rightInputVariable)->Data();

        // Allocate input gradient Value if needed
        auto& inputGradientValue = backPropagatedGradientValuesForInputs[leftInputVariable];
        if (inputGradientValue == nullptr)
            inputGradientValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(DataType::Float, leftInputVariable.Shape(), state->Device()));

        auto rootGradientData = rootGradientValues.at(this->Output())->Data();
        auto inputGradientData = inputGradientValue->Data();

        MatrixMultiply(rootGradientData, rightInputData, inputGradientData, /*transposeRight =*/ true);
    }

    const std::wstring& OpName() const override
    {
        static const std::wstring opName = L"BinMul2A1BOp";
        return opName;
    }

    size_t CurrentVersion() const override { NOT_IMPLEMENTED; }
    const Dictionary Attr;
    const wchar_t* m_key = L"m";
    const wchar_t* n_key = L"n";
    const wchar_t* k_key = L"k";
    const wchar_t* B_key = L"B_values";
    int m;
    int n;
    int k;
    int64_t *binary_B;
    const float *B_data;
    BinGemm2A1B *Executor;

    void InferOutputs(std::vector<Variable>& outputs) override
    {
        auto leftOperand = Inputs()[0];
        auto rightOperand = Inputs()[1];

        if (leftOperand.Shape().Rank() != 2)
            std::runtime_error("Left operand must be 2D");

        if (rightOperand.Shape().Rank() != 1)
            std::runtime_error("Right operand must be 1D");

        if (!leftOperand.DynamicAxes().empty())
            std::runtime_error("Left operand must not have dynamic axes (i.e. should not be minibatch data, but be a Parameter of fixed size)");

        outputs.push_back(OutputVariable(NDShape({ leftOperand.Shape()[0] }), leftOperand.GetDataType(), rightOperand.DynamicAxes()));
    }
};
