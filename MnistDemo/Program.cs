using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;

namespace OnnxInferenceDemo
{
    class Program
    {
        static void Main(string[] args)
        {
            string modelName = "model.onnx";
            if (args.Length > 0)
            {
                modelName = args[0];
            }
            using var session = new InferenceSession(modelName);

            // Example input: a single sample with 64 features (replace with actual values)
            double[] inputData = new double[64];
            for (int i = 0; i < inputData.Length; i++)
                inputData[i] = i / 64.0;  // Example values between 0 and 1

            // Create a tensor with shape [1, 64]
            var tensor = new DenseTensor<double>(inputData, new[] { 1, 64 });

            // Name must match what Netron shows ("X")
            var input = NamedOnnxValue.CreateFromTensor("X", tensor);

            // Run inference
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(new[] { input });

            // Retrieve outputs
            var label = results.First(r => r.Name == "label").AsTensor<long>().ToArray();
            var probabilities = results.First(r => r.Name == "probabilities").AsTensor<double>().ToArray();

            // Output
            Console.WriteLine($"Predicted label: {label[0]}");

            Console.WriteLine("Class probabilities:");
            for (int i = 0; i < probabilities.Length; i++)
                Console.WriteLine($"Class {i}: {probabilities[i]:F4}");
        }
    }
}
