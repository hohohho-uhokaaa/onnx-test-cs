using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

// 1. 各種ファイルのパス定義
string modelPath = "mnist-8.onnx";

string[] imagePaths = new string[]
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
    "9.jpg"
};

foreach (var imagePath in imagePaths)
{
    // 2. ImageSharpでJPG画像を読み込み、AIの仕様（28x28、背景黒/文字白）に一発変換
    using var image = Image.Load<L8>(imagePath);
    image.Mutate(x => x.Resize(28, 28).Invert());

    // 3. 28x28マトリクス（1次元配列）へピクセル値を0.0〜1.0に正規化して詰め込む
    float[] pixelData = new float[28 * 28];
    int idx = 0;
    image.ProcessPixelRows(acc =>
    {
        for (int y = 0; y < acc.Height; y++)
            foreach (var pixel in acc.GetRowSpan(y))
                pixelData[idx++] = pixel.PackedValue / 255f;
    });

    // 4. ONNXの入力プロトコル（[1, 1, 28, 28] の4次元テンサー）に変換
    var inputTensor = new DenseTensor<float>(pixelData, [1, 1, 28, 28]);
    var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("Input3", inputTensor) };

    // 5. ONNXの計算エンジンを起動して推論実行
    using var session = new InferenceSession(modelPath);
    using var results = session.Run(inputs);

    // ==========================================
    // 6. 結果の解析（ソフトマックス関数による確率変換）
    // ==========================================

    // ONNXから出てきた「生スコア（ロジット）」の10個の配列を取得
    float[] rawOutput = results.First().AsEnumerable<float>().ToArray();

    // 各スコアの指数関数（eのx乗）の合計を計算する
    // ※ rawOutput の各値を Math.Exp に通して、その総和（分母）を出す
    double expSum = rawOutput.Select(x => Math.Exp(Convert.ToDouble(x))).Sum();

    // 各部屋の数値を「その部屋のExp値 ÷ 全体の合計（分母）」に変換して、
    // 合計が1.0（100%）に収まる確率の配列を作る
    float[] probabilities = rawOutput.Select(x => (float)(Math.Exp(Convert.ToDouble(x)) / expSum)).ToArray();

    // 最も確率（％）が高かった部屋の「番号（インデックス）」と「その確率」を特定
    float maxProbability = probabilities.Max();
    int resultDigit = Array.IndexOf(probabilities, maxProbability);

    // 画面にパーセント表示（0.0〜100.0%）で出力
    Console.WriteLine($"\n【AI判定結果】 画像に書かれた数字は 「 {resultDigit} 」 です！ (確信度: {maxProbability * 100:F1}% image: {imagePath})");
}