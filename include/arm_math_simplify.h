/*
 * @Description: 
 * @Author: notplus
 * @Date: 2021-03-14 15:34:44
 * @LastEditors: notplus
 * @LastEditTime: 2021-03-14 15:43:38
 */

#include <stdint.h>

/**
   * @brief Error status returned by some functions in the library.
   */

typedef enum
{
    ARM_MATH_SUCCESS = 0,         /**< No error */
    ARM_MATH_ARGUMENT_ERROR = -1, /**< One or more arguments are incorrect */
    ARM_MATH_LENGTH_ERROR = -2,   /**< Length of data buffer is incorrect */
    ARM_MATH_SIZE_MISMATCH = -3,  /**< Size of matrices is not compatible with the operation */
    ARM_MATH_NANINF = -4,         /**< Not-a-number (NaN) or infinity is generated */
    ARM_MATH_SINGULAR = -5,       /**< Input matrix is singular and cannot be inverted */
    ARM_MATH_TEST_FAILURE = -6    /**< Test Failed */
} arm_status;

/**
   * @brief 8-bit fractional data type in 1.7 format.
   */
typedef int8_t q7_t;

/**
   * @brief 16-bit fractional data type in 1.15 format.
   */
typedef int16_t q15_t;

/**
   * @brief 32-bit fractional data type in 1.31 format.
   */
typedef int32_t q31_t;

/**
   * @brief 64-bit fractional data type in 1.63 format.
   */
typedef int64_t q63_t;

/**
   * @brief 32-bit floating-point type definition.
   */
typedef float float32_t;

/**
   * @brief 64-bit floating-point type definition.
   */
typedef double float64_t;

/**
   * @brief Instance structure for the floating-point matrix structure.
   */
typedef struct
{
    uint16_t numRows; /**< number of rows of the matrix.     */
    uint16_t numCols; /**< number of columns of the matrix.  */
    float32_t *pData; /**< points to the data of the matrix. */
} arm_matrix_instance_f32;

/**
   * @brief Floating-point matrix multiplication
   * @param[in]  pSrcA  points to the first input matrix structure
   * @param[in]  pSrcB  points to the second input matrix structure
   * @param[out] pDst   points to output matrix structure
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */
arm_status arm_mat_mult_f32(
    const arm_matrix_instance_f32 *pSrcA,
    const arm_matrix_instance_f32 *pSrcB,
    arm_matrix_instance_f32 *pDst);
