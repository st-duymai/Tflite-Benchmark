package org.tensorflow.lite.examples.androidbenchmark.model

import org.tensorflow.lite.examples.lib_interpreter.InputDataType

data class TfliteModel(
    val name: String,
    val path: String,
    val inputWidth: Int,
    val inputHeight: Int,
    val inputDataType: InputDataType,
    val modelType: ModelType
)
