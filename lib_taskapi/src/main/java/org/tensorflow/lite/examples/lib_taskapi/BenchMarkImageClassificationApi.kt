package org.tensorflow.lite.examples.lib_taskapi

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.vision.classifier.ImageClassifier

class BenchMarkImageClassificationApi(private val objectDetector: ImageClassifier) : Benchmark {

    companion object {

        private const val NUM_CPU_THREAD = 4

        fun create(
            context: Context,
            modelFile: String,
            isUseGpu: Boolean
        ): BenchMarkImageClassificationApi {
            val options = ImageClassifier.ImageClassifierOptions.builder()
                .setBaseOptions(BaseOptions.builder().apply {
                    if (isUseGpu) {
                        useGpu()
                    } else {
                        setNumThreads(NUM_CPU_THREAD)
                    }
                }.build())
                .build()
            val objectDetector = ImageClassifier.createFromFileAndOptions(
                context, modelFile, options
            )
            return BenchMarkImageClassificationApi(objectDetector)
        }
    }

    override fun benchmark(bitmap: Bitmap): Long {
        val tensor = TensorImage.fromBitmap(bitmap)
        val startTime = System.nanoTime()
        objectDetector.classify(tensor)
        return (System.nanoTime() - startTime) / 1000000 // convert nanosecond to milisecond
    }

    override fun close() {
        objectDetector.close()
    }
}
