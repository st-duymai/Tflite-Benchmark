package org.tensorflow.lite.examples.lib_interpreter

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class BenchmarkInterpreter(
    private val interpreter: Interpreter,
    private val width: Int,
    private val height: Int,
    private val dataType: DataType
) : Benchmark {
    private val outputShape = interpreter.getOutputTensor(0).shape()

    companion object {
        private const val NUM_CPU_THREAD = 4
        fun create(
            context: Context,
            filePath: String,
            width: Int,
            height: Int,
            dataType: InputDataType,
            isUseGpu: Boolean
        ): BenchmarkInterpreter {
            val byteBuffer = FileUtil.loadMappedFile(context, filePath)
            val compatList = CompatibilityList()
            val options = Interpreter.Options().apply {
                if (isUseGpu) {
                    val delegateOptions = compatList.bestOptionsForThisDevice
                    addDelegate(GpuDelegate(delegateOptions))
                } else {
                    numThreads = NUM_CPU_THREAD
                }
            }
            val interpreter = Interpreter(byteBuffer, options)
            return BenchmarkInterpreter(
                interpreter, width, height, when (dataType) {
                    InputDataType.UINT8 -> DataType.UINT8
                    else -> DataType.FLOAT32
                }
            )
        }
    }

    override fun benchmark(bitmap: Bitmap): Long {
        val startTime = System.nanoTime()
        val inputTensor = prepareImageTensor(bitmap)
        val outputTensor = TensorBuffer.createFixedSize(outputShape, DataType.FLOAT32)
        interpreter.run(inputTensor?.buffer, outputTensor.buffer.rewind())
        return (System.nanoTime() - startTime) / 1000000 // convert nanosecond to milisecond
    }

    private fun prepareImageTensor(bitmap: Bitmap): TensorImage? {
        val imageProcessor = ImageProcessor.Builder().apply {
            add(ResizeOp(width, height, ResizeOp.ResizeMethod.BILINEAR))
        }.build()
        val tensorImage = TensorImage(dataType)
        tensorImage.load(bitmap)
        return imageProcessor.process(tensorImage)
    }

    override fun close() {
        interpreter.close()
    }
}
