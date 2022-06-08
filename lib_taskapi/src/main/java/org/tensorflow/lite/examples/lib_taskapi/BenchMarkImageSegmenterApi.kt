package org.tensorflow.lite.examples.lib_taskapi

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.vision.segmenter.ImageSegmenter

class BenchMarkImageSegmenterApi(private val objectDetector: ImageSegmenter) : Benchmark {

    companion object {
        fun create(context: Context, modelFile: String): BenchMarkImageSegmenterApi {
            val options = ImageSegmenter.ImageSegmenterOptions.builder()
                .setBaseOptions(BaseOptions.builder().useGpu().build())
                .build()
            val objectDetector = ImageSegmenter.createFromFileAndOptions(
                context, modelFile, options
            )
            return BenchMarkImageSegmenterApi(objectDetector)
        }
    }

    override fun benchmark(bitmap: Bitmap): Long {
        val startTime = System.nanoTime()
        objectDetector.segment(TensorImage.fromBitmap(bitmap))
        return (System.nanoTime() - startTime) / 1000000 // convert nanosecond to milisecond
    }

    override fun close() {
        objectDetector.close()
    }
}
