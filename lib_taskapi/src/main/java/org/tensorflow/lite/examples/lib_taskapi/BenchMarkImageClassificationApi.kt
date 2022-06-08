package org.tensorflow.lite.examples.lib_taskapi

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.vision.classifier.ImageClassifier

class BenchMarkImageClassificationApi(private val objectDetector: ImageClassifier) : Benchmark {

    companion object {
        fun create(context: Context, modelFile: String): BenchMarkImageClassificationApi {
            val options = ImageClassifier.ImageClassifierOptions.builder()
                .setBaseOptions(BaseOptions.builder().useGpu().build())
                .build()
            val objectDetector = ImageClassifier.createFromFileAndOptions(
                context, modelFile, options
            )
            return BenchMarkImageClassificationApi(objectDetector)
        }
    }

    override fun benchmark(bitmap: Bitmap): Long {
        val startTime = System.nanoTime()
        objectDetector.classify(TensorImage.fromBitmap(bitmap))
        return (System.nanoTime() - startTime) / 1000000 // convert nanosecond to milisecond
    }

    override fun close() {
        objectDetector.close()
    }
}
