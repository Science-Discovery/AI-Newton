from typing import List, Literal
import numpy as np
import sympy as sp
from aiphy.core import NormalData, ExpData, is_conserved_mean_and_std
# NOTE: dtype should be considered?


def is_conservation(mean, std, alpha=0.05):
    return is_conserved_mean_and_std(mean, std, alpha)


# def calc_linear_combination(coe: np.ndarray,
#                             lst_normaldata: list[NormalData]) -> NormalData:
#     """
#     Calculate the dot product of the numpy array 'coe' with the list of numpy arrays 'lst_vars'.

#     Args:
#         coe (numpy array): The array with the same length as 'lst'.
#         lst_normaldata (list of NormalData): A list of NormalData objects.

#     Returns:
#         numpy array: The dot product result.

#     """
#     # Check if the length of 'coe' matches the length of 'lst_vars'
#     if len(coe) != len(lst_normaldata):
#         raise ValueError("The length of 'coe' must match the length of 'lst_vars'.")
#     # Initialize the result as zeros with the same shape as lst_vars[0]
#     # result = np.zeros_like(lst_normaldata[0])
#     result = NormalData.from_elem(0, 0, n=lst_normaldata[0].n, repeat_time=lst_normaldata[0].repeat_time)
#     # Calculate the dot product
#     for i in range(len(coe)):
#         coe_temp = NormalData.from_elem(coe[i], 0, n=lst_normaldata[i].n, repeat_time=lst_normaldata[i].repeat_time)
#         # result += coe[i] * lst_normaldata[i]
#         result = result + coe_temp * lst_normaldata[i]
#     return result


def calc_linear_combination(coe, lst_vars):
    """
    Calculate the dot product of the numpy array 'coe' with the list of numpy arrays 'lst_vars'.

    Args:
        coe (numpy array): The array with the same length as 'lst'.
        lst_vars (list of numpy arrays): A list of numpy arrays.

    Returns:
        numpy array: The dot product result.

    """
    # Check if the length of 'coe' matches the length of 'lst_vars'
    if len(coe) != len(lst_vars):
        raise ValueError("The length of 'coe' must match the length of 'lst_vars'.")
    # Initialize the result as zeros with the same shape as lst_vars[0]
    result = np.zeros_like(lst_vars[0])
    # Calculate the dot product
    for i in range(len(coe)):
        result += coe[i] * lst_vars[i]
    return result


def calc_weights_from_errors(errors: list[list] | list[np.ndarray]) -> np.ndarray:
    errors = np.array(errors, dtype=np.float64)
    error_squared_sum = np.sum(errors**2, axis=0)
    error_squared_sum_inv = 1 / error_squared_sum  # NOTE: regularization may be needed
    normalization_factor = np.sum(error_squared_sum_inv)
    weights = error_squared_sum_inv / normalization_factor
    weights = np.array(weights, dtype=np.float64)  # Shape = (n_t,)
    return weights


def check_zero_coefficients(coes, value, error, reduced=False, specific_row=None):
    """
    Check and potentially set coefficients to zero if they are effectively zero based on whether the correspoding
    linear combinations give constants.

    Args:
        coes (list of lists): List of coefficient lists, one gives rise to a constant.
        value (list of numpy arrays): List of numpy arrays representing values of all monomials.
        error (list of numpy arrays): List of numpy arrays representing errors of all monomials.
        reduced (bool, optional): Whether the coefficients have been performed row reduce. Default is False.

    Returns:
        list of lists: Updated coefficient lists with some elements potentially set to zero.

    Raises:
        ValueError: If the length of any one of those oefficients doesn't match with values or errors.
    """
    arg_zero_array = []
    res = np.copy(np.array(coes))
    for coe in res:
        if specific_row is not None and res.index(coe) not in specific_row:
            arg_zero_array.append([])
            continue
        # Check if the length of coefficients doesn't match with values or errors
        if len(coe) != len(value) or len(coe) != len(error):
            raise ValueError("Length of coefficients doesn't match with values or errors.")
        order_ralative_small = 1e-2  # The order used to determine what's "relative small"
        # Set critical value for different situations
        if reduced:
            critical = max(np.abs(coe)) * order_ralative_small
        else:
            # critical = 1/np.sqrt(len(coe)) * order_ralative_small
            critical = max(np.abs(coe)) * order_ralative_small
        # Record the arguments that the corresponding elements that are actually zero in this row
        arg_zero_this_row = []
        for arg, term in enumerate(coe):
            if np.abs(term) != 0 and np.abs(term) < critical:
                coe_temp = np.copy(coe)
                # ？Is this necessary?
                # coe_temp[arg_zero_this_row] = 0
                coe_temp[arg] = 0
                flag = is_conservation(calc_linear_combination(coe_temp, value),  # NOTE:
                                       error_estimation_for_regression(coe_temp, error))
                if flag:
                    arg_zero_this_row.append(arg)
        arg_zero_array.append(arg_zero_this_row)
    for i in range(len(res)):
        res[i][arg_zero_array[i]] = 0
    return res


def error_estimation_for_regression(coe, lst_errors):
    # Check if the length of 'coe' matches the length of 'lst_vars'
    if len(coe) != len(lst_errors):
        raise ValueError("The length of 'coe' must match the length of 'lst_errors'.")
    # Initialize the result as zeros with the same shape as lst_errors[0]
    sigma2 = np.zeros_like(lst_errors[0])
    # Calculate squared deviation
    for i in range(len(coe)):
        sigma2 += (coe[i] * lst_errors[i]) ** 2
    # Get standard deviation
    sigma = np.sqrt(sigma2)
    return sigma


def first_nonzero_index_numpy(arr):
    nonzero_indices = np.nonzero(arr)[0]
    return nonzero_indices[0] if nonzero_indices.size > 0 else None


def merge_lists(lst: list, dim: int = 0) -> list:
    """
    Merge a list of lists into a single list.
    Each sub list corresponds to a specifc experiment. It contains time series arrays
    of the same length. Concatenation is performed along the specified dimension.

    Args:
        lst (list): A list of lists.
        dim (int, optional): The dimension along which to merge the lists (default is 0).

    Returns:
        list: A list containing all elements from the input list of lists.

    Raises:
        ValueError: If the dimension is out of bounds.
    """
    # Convert lst to a numpy array
    lst = np.array(lst)
    # Check if the dimension is out of bounds
    if dim < 0 or dim >= len(lst[0].shape):
        raise ValueError("Dimension out of bounds.")
    # Merge the lists along the specified dimension
    merged_list = np.concatenate(lst, axis=dim)
    # Convert the result back to a list
    merged_list = list(merged_list)
    return merged_list


def multivariate_linear_regression(list_of_arrays: list[np.ndarray],
                                   list_of_errors: list[np.ndarray],) -> list[np.ndarray]:
    """
    Perform multivariate linear regression on a list of arrays with corresponding errors.

    Args:
    - list_of_arrays (list of numpy arrays): A list of numpy arrays representing the means of the data.
    - list_of_errors (list of numpy arrays): A list of numpy arrays representing the errors of the data.

    Returns:
    - a list of weights for each array that satisfy the conservation law
    """
    list_of_arrays = np.array(list_of_arrays, dtype=np.float64)
    # len = n_terms, shape = (n_t * num_exp,)
    list_of_weights = calc_weights_from_errors(list_of_errors)

    # Solve the eigenvalue problem of the covariance matrix
    cov_matrix = np.cov(list_of_arrays, aweights=list_of_weights)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    except np.linalg.LinAlgError:
        print(cov_matrix, '\n', list_of_arrays, '\n', list_of_weights)
        raise np.linalg.LinAlgError("Eigenvalue decomposition failed.")

    # Select the linear combinations that satisfy the conservation law
    sol = [eigenvectors[:, i] for i in range(len(eigenvalues))
           if is_conservation(calc_linear_combination(eigenvectors[:, i], list_of_arrays),
                              error_estimation_for_regression(eigenvectors[:, i], list_of_errors))]
    return sol


def normalize_array(array: np.array, scheme: Literal['square', 'first'] = 'square') -> np.ndarray:
    """
    Normalize a numpy array according to the specified scheme.

    Args:
        array (np.array): The input array to be normalized.
        scheme (Literal['square', 'first'], optional): The normalization scheme (default is 'square').
            - 'square': Normalize by dividing each element by the square root of the sum of
                        squares of all elements.
            - 'first': Normalize by dividing each element by the first element.

    Returns:
        np.ndarray: The normalized array.

    Raises:
        ValueError: If the scheme is invalid.
        ValueError: If the input array is not a 1D or 2D numpy array.
    """
    # Check if the scheme is valid
    if scheme not in ['square', 'first']:
        raise ValueError("Invalid normalization scheme.")
    # Scheme: 'square'
    if scheme == 'square':
        if len(array.shape) == 1:
            normalized_array = array / np.sqrt(np.sum(array ** 2))
        elif len(array.shape) == 2:
            normalized_array = array / np.sqrt(np.sum(array ** 2, axis=1, keepdims=True))
        else:
            raise ValueError("normalize_array: Input array must be a 1D or 2D numpy array.")
    # Scheme: 'first'
    elif scheme == 'first':
        if len(array.shape) == 1:
            not_zero_pos = first_nonzero_index_numpy(array)
            if not_zero_pos:
                normalized_array = array / array[not_zero_pos]
            else:
                normalized_array = array
        elif len(array.shape) == 2:
            normalized_array = []
            for arr in array:
                not_zero_pos = first_nonzero_index_numpy(arr)
                if not_zero_pos:
                    normalized_array.append(arr/arr[not_zero_pos])
                else:
                    normalized_array.append(arr)
            normalized_array = np.array(normalized_array)
        else:
            raise ValueError("normalize_array: Input array must be a 1D or 2D numpy array.")
    return normalized_array


def rationalize_array(array, tolerance=0.1):
    # NOTE: To use this function, the normalization scheme must be 'first'?
    if np.all([isinstance(arr, np.ndarray) for arr in array]):
        return [rationalize_array(arr, tolerance) for arr in array]
    elif np.all([isinstance(arr, float) for arr in array]):
        # Devide the biggest element in the array
        array = array / max(np.abs(array))
        res = [sp.nsimplify(arr, rational=True, tolerance=tolerance, full=True) for arr in array]
        denominator_set = set([fraction.q for fraction in res])
        if len(denominator_set) == 1:
            common_denominator = list(denominator_set)[0]
        elif len(denominator_set) == 2:
            common_denominator = sp.lcm(*denominator_set)
        else:
            denominators = list(denominator_set)
            common_denominator = sp.lcm(denominators[0], denominators[1])
            for denominator in denominators[2:]:
                common_denominator = sp.lcm(common_denominator, denominator)
        res = [int(fraction * common_denominator) for fraction in res]
        return res
    else:
        raise ValueError("elements out of float can't be rationalized")


def remove_useless(ids_remain: list, means: list, errors: list, coes_final: np.ndarray) -> tuple:
    """
    Remove useless monomials, zero equations, and repeated equations from the input data.

    Args:
        monos (list): List of monomials(expressions).
        means (list): List of means.
        errors (list): List of errors.
        coes_final (np.array): Array of coefficients.

    Returns:
        tuple: A tuple containing the updated monomials, means, errors, coefficients and
               the flag representing is there any monomial removed.
    """
    # Remove useless monomials
    # Iterate over all monomials and check if their coefficients are all zero in all equations
    flag = False  # Flag representing is there any monomial removed
    useless_mono_indices = select_useless_mono(coes_final)
    if useless_mono_indices:
        flag = True
    ids_remain = [ids_remain[i] for i in range(len(ids_remain)) if i not in useless_mono_indices]
    means = [means[i] for i in range(len(means)) if i not in useless_mono_indices]
    errors = [errors[i] for i in range(len(errors)) if i not in useless_mono_indices]
    coes_final = np.delete(coes_final, useless_mono_indices, axis=1)
    # Remove zero equations
    zero_eq_indices = select_zero_eq(coes_final)
    coes_final = np.delete(coes_final, zero_eq_indices, axis=0)
    # Remove repeated equations
    # 'Reapeated equations' means that the coefficients of all monomials are the same up to a small relative error(i.e. 1e-2) in two equations
    repeated_eq_indices = select_repeated_eq(coes_final)
    coes_final = np.delete(coes_final, repeated_eq_indices, axis=0)
    # Final check if the length of ids_remain, means and errors are the same
    if len(ids_remain) != len(means) or len(ids_remain) != len(errors):
        raise ValueError("The length of ids_remain, means and errors must be the same.")
    return ids_remain, means, errors, coes_final, flag


def row_reduce(matrix: list | np.ndarray, value, error) -> np.ndarray:
    """
    Perform row reduction (Gaussian elimination) on a 2D numpy array representing a matrix.

    Args:
        matrix (np.ndarray): The input matrix to be row-reduced.

    Returns:
        np.ndarray: The row-reduced matrix.

    Raises:
        ValueError: If the input is not a 2D numpy array.
    """
    # Check if the input matrix is an instance of a list, if so, convert it into np.array
    if isinstance(matrix, list):
        matrix = np.array(matrix)
    if len(matrix.shape) != 2:
        raise ValueError("row_reduce: Input matrix must be a 2D numpy array.")
    # Get the number of rows and columns in the matrix
    num_rows, num_cols = matrix.shape
    # Create a copy of the input matrix to avoid modifying the original one
    result_matrix = np.copy(matrix)
    # Set the elements that are effectively zero to zero
    result_matrix = check_zero_coefficients(coes=result_matrix, value=value, error=error, reduced=True)
    # Normalize each row
    result_matrix = normalize_array(result_matrix)
    # Perform Gaussian elimination column-by-column from left to right
    for col in range(num_cols):
        # print(result_matrix)  # For step-by-step check
        # Find the pivot element (the first nonzero element) in the current column
        pivot_row = -1
        for row in range(col, num_rows):
            if result_matrix[row, col] != 0:  # Those small elements who are actually zero has to be set to zero in advance
                pivot_row = row
                break
        if pivot_row == -1:
            # No pivot element found, move to the next column
            continue
        # Swap the pivot row with the current row (if necessary)
        if pivot_row != col:
            result_matrix[[col, pivot_row]] = result_matrix[[pivot_row, col]]
        # Make the pivot element 1
        result_matrix[col] /= result_matrix[col, col]
        # Eliminate other elements in the current column
        for row in range(num_rows):
            if row == col:
                continue
            factor = result_matrix[row, col]
            result_matrix[row] -= factor * result_matrix[col]
    # Perform further Gaussion elimination column-by-column from right to left, try to make the matrix's upper left part diagonal
    for col in range(min(num_rows, num_cols)-1, -1, -1):
        # print(result_matrix)  # For step-by-step check
        # Check if the diagonal element is zero
        if result_matrix[col, col] == 0:
            # If so, continue to the next column
            continue
        # If not, eliminate other elements in the current column
        for row in range(col):
            factor = result_matrix[row, col]
            result_matrix[row] -= factor * result_matrix[col]
    # Normalize coeffients
    result_matrix = normalize_array(result_matrix)
    # Set the elements that are effectively zero to zero
    result_matrix = check_zero_coefficients(coes=result_matrix, value=value, error=error, reduced=True)
    # Normalize again
    result_matrix = normalize_array(result_matrix)
    # Perform further row reduce on remaining columns
    # result_matrix = lower_rank_simplification(result_matrix, value, error)
    return result_matrix


def row_reduce_model(coes, ids_remain, lst_means, lst_errors):
    """
    Perform row reduction on the coefficients matrix and remove useless rows.

    Args:
        coes (list): The coefficients matrix.
        lst_means (list): The list of means.
        lst_errors (list): The list of errors.

    Returns:
        tuple: A tuple containing the row-reduced coefficients matrix, the updated list of monomials,
                the updated list of means, and the updated list of errors.
    """
    flag = True
    coes_final = coes  # Initialize the final coefficients matrix as the input one
    while flag:
        if coes:
            coes_final = row_reduce(matrix=coes_final, value=lst_means, error=lst_errors)
            ids_remain, lst_means, lst_errors, coes_final, flag = remove_useless(ids_remain, lst_means, lst_errors, coes_final)
        # If the coefficients matrix is empty, break the loop
        else:
            coes_final, ids_remain, lst_means, lst_errors = [], [], [], []
            break
    return coes_final, ids_remain, lst_means, lst_errors


def select_useless_mono(coes_final: np.ndarray) -> list:
    useless_mono_indices = []
    for i in range(len(coes_final[0])):
        if np.all(coes_final[:, i] == 0):
            useless_mono_indices.append(i)
    return useless_mono_indices


def select_zero_eq(coes_final: np.ndarray) -> list:
    zero_eq_indices = []
    for i in range(len(coes_final)):
        if np.all(coes_final[i] == 0):
            zero_eq_indices.append(i)
    return zero_eq_indices


def select_repeated_eq(coes_final: np.ndarray) -> list:
    repeated_eq_indices = []
    for i in range(len(coes_final)):
        for j in range(i+1, len(coes_final)):
            # Calculate the relative error
            relative_error = np.abs(np.abs(coes_final[i]) - np.abs(coes_final[j])) / (np.abs(coes_final[i]) + 1e-10)
            # If the relative error is small enough, the two equations are considered to be the same
            if np.all(relative_error < 1e-2):
                repeated_eq_indices.append(j)
    return repeated_eq_indices


def subtract_weighted_mean(lst_means: list, lst_errors: list) -> list:
    """
    Subtract the weighted mean from each mean array in the list.
    Weights are calculated from the errors via calc_weights_from_errors().
    The result is centered around zero.

    Args:
        lst_means (list): A list of mean arrays.
        lst_errors (list): A list of error arrays.

    Returns:
        list: A list containing the weighted mean subtracted from each mean array in the input list.

    Raises:
        ValueError: If the lengths of lst_means and lst_errors are not the same.
        ValueError: If the lengths of lst_means[0] and lst_errors[0] are not the same.
    """
    # Check if the lengths of lst_means and lst_errors are the same
    if len(lst_means) != len(lst_errors):
        raise ValueError("The lengths of lst_means and lst_errors must be the same.")
    # Check if the lengths of all arrays in lst_means and lst_errors are the same
    if len(set([len(lst_means[i]) for i in range(len(lst_means))])) != 1 or \
       len(set([len(lst_errors[i]) for i in range(len(lst_errors))])) != 1:
        raise ValueError("The lengths of all arrays in lst_means and lst_errors must be the same.")

    # Calculate the weights from the errors
    arr_weights = calc_weights_from_errors(lst_errors)  # Shape = (n_t,)
    # Calculate the weighted mean of each mean array
    lst_weighted_means = [np.average(lst_means[i], weights=arr_weights) for i in range(len(lst_means))]
    # Subtract the weighted mean from each mean array
    lst_means_subtracted = [lst_means[i] - lst_weighted_means[i] for i in range(len(lst_means))]
    # len = len(lst_means) = len(lst_errors), shape = (n_t,)
    return lst_means_subtracted


def compensate_zero_coefficients(coes, ids_remain, num_terms_origin):
    """
    Iterate over all non-negative integers up to num_terms_origin.
    If an integer is not in ids_remain, insert a zero coefficient at the corresponding position in coes.
    """
    if not coes:
        return list([np.array([0] * num_terms_origin, dtype=np.int32)])
    coes = np.array(coes, dtype=np.int32)
    for i in range(num_terms_origin):
        if i not in ids_remain:
            coes = np.insert(coes, i, 0, axis=1)
    return list(coes)


def pca_regression(an: List[NormalData]) -> List[List[int]]:
    """
    Perform PCA regression on a list of NormalData objects.

    Args:
        an: A list of NormalData objects. Each object corresponds to a monomial.
            len(an) = n_terms, shape = (repeat_time, n_t)

    NOTE: Monomials and their corresponding data should be sorted in adequate order
    before calling this function.
    """
    num_terms_origin = len(an)

    # Process the input data
    lst_means = [np.array(a.mean, dtype=np.float64) for a in an]
    # len = n_terms, shape = (n_t * num_exp,)
    lst_errors = [np.array(a.std, dtype=np.float64) for a in an]

    badpts = set()
    for a in an:
        badpts |= a.badpts
    for i in range(len(lst_means)):
        lst_means[i] = np.delete(lst_means[i], list(badpts))
        lst_errors[i] = np.delete(lst_errors[i], list(badpts))

    ids_remain = [i for i in range(len(lst_means))]  # The indices of monomials that remain after row reduction

    # Perform multivariate linear regression to get raw coefficients
    try:
        coes = multivariate_linear_regression(lst_means, lst_errors)  # len = n_sols, shape = (n_terms_remain,)
    except Exception:
        coes = []

    # Row reduce the coefficients matrix and remove useless monomials
    coes, ids_remain, lst_means, lst_errors = row_reduce_model(coes, ids_remain, lst_means, lst_errors)

    # Rationalize the coefficients
    coes = [rationalize_array(coe) for coe in coes]

    # Add those zero coefficients back
    coes = compensate_zero_coefficients(coes, ids_remain, num_terms_origin)

    return coes


from typing import Tuple
from aiphy import DataStruct, Exp


def search_relations_by_pca(ds: DataStruct) -> List[Tuple[Exp, ExpData]]:
    atom_list: List[Exp] = []
    data_list: List[ExpData] = []
    # print(ds)
    for atom in ds.data_keys:
        data = ds.fetch_data_by_key(atom)
        if not data.is_normal:
            continue
        atom_list.append(Exp.Atom(atom))
        data_list.append(data)

    res = []
    for i in range(len(atom_list)):
        for j in range(i):
            data: ExpData = data_list[i] * data_list[j]
            if data.is_const:
                res.append((atom_list[i] * atom_list[j], data))
            data: ExpData = data_list[i] / data_list[j]
            if data.is_const:
                res.append((atom_list[i] / atom_list[j], data))

    if len(data_list) < 2:
        return res

    info_list: List[Tuple[Exp, ExpData, NormalData]] = []
    for i in range(len(data_list)):
        mean = data_list[i].calc_mean
        if mean is None:
            continue
        normaldata = data_list[i].normal_data - NormalData.from_const_data(mean, data_list[i].n, data_list[i].repeat_time)
        info_list.append((atom_list[i], data_list[i], normaldata))

    result = pca_regression(list(map(lambda x: x[2], info_list)))
    for coe in result:
        exp = Exp.Number(0)
        data: ExpData = ExpData.from_const(0, 0)
        assert len(coe) == len(info_list)
        for i in range(len(coe)):
            if coe[i] != 0:
                exp = exp + info_list[i][0] * Exp.Number(coe[i])
                data = data + info_list[i][1] * ExpData.from_const(coe[i], 0)
        if exp.type == 'Number':
            continue
        if data.is_conserved:
            # print(coe)
            res.append((exp, data))
            # print("debug in pca", exp, data)
        else:
            # print(f"debug in pca, {exp}, Not conserved!")
            pass
    return res
