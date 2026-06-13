// after qwen2.5-coder:7b prossessing, 2024-06-17
// add comments, refactor to method, and add loop for multiple images by qwen2.5-coder:7B

// Program.cs
using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace ImageClassificationAI
{
    class Program
    {
        private static readonly string modelPath = "mnist-8.onnx";
        private static readonly List<string> imagePaths = new()
        {
            "0.jpg",
            "1.jpg",
            "2.jpg",
            "3.jpg",
            "4.jpg",
            "5.jpg",
            "6.jpg",
            "7.jpg",
            "8.jpg",
            "9.jpg",
            "400dpi-28x28x-8.png",
        };

        static void Main(string[] args)
        {
            // ONNXの計算エンジンを起動（1回だけ作成して再利用）
            using var session = new InferenceSession(modelPath);

            // 各画像ファイルを処理する
            foreach (var imagePath in imagePaths)
            {
                ProcessImage(imagePath, session);
            }
        }

        private static void ProcessImage(string imagePath, InferenceSession session)
        {
            // ImageSharpを使用して画像を読み込み、28x28ピクセルのサイズにリサイズし、背景黒文字白にインバートする
            using var image = Image.Load<L8>(imagePath);
            image.Mutate(x => x.Resize(28, 28).Invert());

            // ピクセルデータを1次元配列に正規化して詰める
            float[] pixelData = new float[28 * 28];
            int idx = 0;
            image.ProcessPixelRows(acc =>
            {
                for (int y = 0; y < acc.Height; y++)
                    foreach (var pixel in acc.GetRowSpan(y))
                        pixelData[idx++] = pixel.PackedValue / 255f; // ピクセル値の正規化
            });

            // ONNXの入力プロトコルに変換する
            var inputTensor = new DenseTensor<float>(pixelData, [1, 1, 28, 28]);
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("Input3", inputTensor) };

            // 推論を行う
            using var results = session.Run(inputs);

            // 推論結果から「生スコア（ロジット）」を取得する
            float[] rawOutput = results.First().AsEnumerable<float>().ToArray();

            // ソフトマックス関数を適用して確率変換を行う
            double expSum = rawOutput.Select(x => Math.Exp(Convert.ToDouble(x))).Sum();
            float[] probabilities = rawOutput.Select(x => (float)(Math.Exp(Convert.ToDouble(x)) / expSum)).ToArray();

            // 確率が最も高いクラスのインデックスを取得する
            float maxProbability = probabilities.Max();
            int resultDigit = Array.IndexOf(probabilities, maxProbability);

            // AI判定結果を画面に出力する
            Console.WriteLine($"\n【AI判定結果】 画像に書かれた数字は 「 {resultDigit} 」 です！ (確信度: {maxProbability * 100:F1}% image: {imagePath})");
        }
    }
}