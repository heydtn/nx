defmodule Nx.BinaryBackend.Matrix do
  @moduledoc false
  use Complex.Kernel

  import Nx.Shared

  def ts(a_data, a_type, a_shape, b_data, b_type, b_shape, output_type, input_opts) do
    # Here's Paulo Valente's proof in the deleted function ts_handle_opts/3.

    # When lower: false, we need the following procedure
    # for reusing the lower_triangular engine:
    #
    # First, we need to reverse both rows and columns
    # so we can turn an upper-triangular matrix
    # into a lower-triangular one.
    # The result will also be reversed in this case.
    #
    # Proof:
    # For a result [x1, x2, x3.., xn] and a row [a1, a2, a4, ..., an]
    # we have the corresponding b = a1 * x1 + a2 * x2 + a3 * x3 + ...+ an * xn
    # Since the addition of a_i * x_i is commutative, by reversing the columns
    # of a, the yielded x will be reversed.
    # Furthermore, if we reverse the rows of a, we need to reverse the rows of b
    # so each row is kept together with it's corresponding result.
    #
    # For example, the system:
    # A = [[a b c], [0 d e], [0 0 f]]
    # b = [b1, b2, b3, b4]
    # which yields x = [x1, x2, x3, x4]
    # is therefore equivalent to:
    # A = [[f 0 0], [e d 0], [c b a]]
    # b = [b4, b3, b2, b1]
    # which yields [x4, x3, x2, x1]

    # For handling left_side: false
    # Let's notate the system matrix as L when it's a lower triangular matrix
    # and U when it's upper triangular
    #
    # To solve X.L = B, we can then transpose both sides:
    # transpose(X.L) = transpose(L).X_t = U.X_t = b_t
    # This equation, in turn, has the same shape (U.X = B) as the one we can solve through
    # applying `ts_handle_lower_opt` properly, which would yield X_t.
    # Transposing the result suffices for yielding the final result.

    opt_left_side = !!input_opts[:left_side]
    opt_lower = !!input_opts[:lower]
    transform_a = input_opts[:transform_a]

    # For :conjugate, we pre-conjugate the matrix elements and treat as :none
    {conjugate_a?, opt_transpose_a} =
      case transform_a do
        :conjugate -> {true, false}
        :transpose -> {false, true}
        _ -> {false, false}
      end

    transpose_a? = opt_left_side == opt_transpose_a
    reverse_a? = opt_left_side == (opt_lower == opt_transpose_a)

    transpose_b? = not opt_left_side and match?({_, _}, b_shape)
    reverse_b? = reverse_a?

    transpose_x? = transpose_b?
    reverse_x? = reverse_b?

    a_matrix =
      a_data
      |> binary_to_matrix(a_type, a_shape)
      |> then(&if(conjugate_a?, do: conjugate_matrix(&1), else: &1))
      |> then(&if(transpose_a?, do: transpose_matrix(&1), else: &1))
      |> then(&if(reverse_a?, do: reverse_matrix(&1), else: &1))

    b_matrix_or_vec =
      case b_shape do
        {_rows, _cols} ->
          b_data |> binary_to_matrix(b_type, b_shape)

        {_rows} ->
          b_data |> binary_to_vector(b_type)
      end
      |> then(&if(transpose_b?, do: transpose_matrix(&1), else: &1))
      |> then(&if(reverse_b?, do: reverse_matrix(&1), else: &1))

    b_shape =
      case {b_shape, transpose_b?} do
        {{rows, cols}, true} -> {cols, rows}
        _ -> b_shape
      end

    result =
      a_matrix
      |> do_ts(b_matrix_or_vec, b_shape)
      |> then(&if(transpose_x?, do: transpose_matrix(&1), else: &1))
      |> then(&if(reverse_x?, do: reverse_matrix(&1), else: &1))

    matrix_to_binary(result, output_type)
  end

  defp do_ts(a_matrix, b_matrix, {_rows, cols}) do
    1..cols//1
    |> Enum.map(fn b_col ->
      b_vector = get_matrix_column(b_matrix, b_col - 1)

      do_ts(a_matrix, b_vector, 0, [])
    end)
    |> transpose_matrix()
  end

  defp do_ts(a_matrix, b_vector, {_}) do
    do_ts(a_matrix, b_vector, 0, [])
  end

  defp do_ts([row | rows], [b | bs], idx, acc) do
    value = Enum.fetch!(row, idx)

    if Complex.abs(value) == 0 do
      raise ArgumentError, "can't solve for singular matrix"
    end

    y = (b - dot_matrix(row, acc)) / value
    do_ts(rows, bs, idx + 1, acc ++ [y])
  end

  defp do_ts([], [], _idx, acc), do: acc

  # Whether a matrix is singular to working precision: some pivot of its LU
  # factorization is no larger than the rounding error allowed for n
  # operations in the type times the sum of the magnitudes of the products
  # subtracted to form it, that is, no larger than the rounding error of its
  # own computation. That covers an exactly zero pivot and the rounding noise
  # an exactly singular matrix can leave in its place, while a pivot that is
  # small because the matrix is graded still counts, and so does a pivot or
  # sum that overflowed the type. The elimination runs in the same type with
  # the same rounding as the solve, after scaling each row by 2 ** -exponent,
  # with the exponents computed as invert/1 computes them. Non-finite input
  # is left to the solve itself.
  def singular?(a_data, a_type, {n, n} = a_shape, exponents) do
    type =
      a_type
      |> Nx.Type.to_floating()
      |> Nx.Type.to_real()

    rows = binary_to_matrix(a_data, a_type, a_shape)

    has_non_finite? =
      rows
      |> List.flatten()
      |> Enum.any?(&non_finite?/1)

    if has_non_finite? do
      false
    else
      epsilon =
        type
        |> Nx.Constants.epsilon()
        |> Nx.to_number()

      tolerance = round_to_type(Nx.Type.rounding_error_multiplier(type, n) * epsilon, type)

      rows
      |> Enum.zip_with(exponents, &scale_row(&1, type, &2))
      |> pivots(type, [])
      |> Enum.any?(fn {magnitude, subtracted} ->
        threshold = threshold(tolerance, subtracted, type)
        not is_number(magnitude) or not is_number(threshold) or magnitude <= threshold
      end)
    end
  end

  # The tolerance times the sum, rounded as invert/1 rounds them in the type:
  # the sum once after adding up the products and the product once more. A
  # pivot within a rounding of the tolerance would otherwise be decided
  # differently here.
  defp threshold(tolerance, subtracted, type) do
    case round_to_type(subtracted, type) do
      rounded_sum when is_number(rounded_sum) -> round_to_type(tolerance * rounded_sum, type)
      overflowed -> overflowed
    end
  end

  defp non_finite?(%Complex{re: re, im: im}), do: is_atom(re) or is_atom(im)
  defp non_finite?(x), do: is_atom(x)

  defp scale_row(row, type, exponent) do
    scale = :math.pow(2, -exponent)
    Enum.map(row, &{round_to_type(&1 * scale, type), 0})
  end

  # The magnitude of each pivot of an LU factorization with partial pivoting,
  # paired with the sum of the magnitudes of the products subtracted to form
  # it, stopping at the first zero pivot. Each entry carries that sum along
  # through the elimination.
  defp pivots([], _type, acc), do: acc

  defp pivots(rows, type, acc) do
    [[{pivot, subtracted} | pivot_row] | rest] = swap_pivot_row(rows)
    magnitude = Complex.abs(pivot)

    if magnitude == 0 do
      [{0.0, subtracted} | acc]
    else
      rest
      |> Enum.map(fn [{x, _} | row] ->
        subtract_row(row, pivot_row, round_to_type(x / pivot, type), type)
      end)
      |> pivots(type, [{magnitude, subtracted} | acc])
    end
  end

  defp subtract_row([{x, subtracted} | xs], [{u, _} | us], multiplier, type) do
    product = round_to_type(multiplier * u, type)

    [
      {round_to_type(x - product, type), subtracted + Complex.abs(product)}
      | subtract_row(xs, us, multiplier, type)
    ]
  end

  defp subtract_row([], [], _multiplier, _type), do: []

  defp round_to_type(%Complex{re: re, im: im}, type),
    do: Complex.new(round_to_type(re, type), round_to_type(im, type))

  defp round_to_type(number, _type) when is_atom(number), do: number
  defp round_to_type(number, {:f, 64}), do: number * 1.0

  defp round_to_type(number, {:f, 32}) do
    case <<number::float-32>> do
      <<rounded::float-32>> -> rounded
      _ when number > 0 -> :infinity
      _ -> :neg_infinity
    end
  end

  defp round_to_type(number, type), do: round_number_to_type(number, type)

  defp round_number_to_type(number, type) do
    match_types [type] do
      <<match!(rounded, 0)>> = <<write!(number, 0)>>
      read!(rounded, 0)
    end
  end

  # The row whose leading entry has the largest magnitude, swapped to the front
  defp swap_pivot_row([[{first, _} | _] = first_row | rest] = rows) do
    first_magnitude = Complex.abs(first)

    {_, index} =
      rest
      |> Enum.with_index(1)
      |> Enum.reduce({first_magnitude, 0}, fn {[{x, _} | _], i}, {max, max_index} ->
        magnitude = Complex.abs(x)
        if magnitude > max, do: {magnitude, i}, else: {max, max_index}
      end)

    if index == 0,
      do: rows,
      else: [Enum.at(rows, index) | List.replace_at(rest, index - 1, first_row)]
  end

  ## Matrix (2-D array) manipulation

  defp dot_matrix([], _), do: 0
  defp dot_matrix(_, []), do: 0

  defp dot_matrix([h1 | _] = v1, [h2 | _] = v2) when not is_list(h1) and not is_list(h2) do
    Enum.zip_reduce(v1, v2, 0, fn x, y, acc -> x * y + acc end)
  end

  defp dot_matrix(m1, m2) do
    # matrix multiplication which works on 2-D lists
    Enum.map(m1, fn row ->
      m2
      |> transpose_matrix()
      |> Enum.map(fn col ->
        Enum.zip_reduce(row, col, 0, fn x, y, acc -> acc + x * Complex.conjugate(y) end)
      end)
    end)
  end

  defp conjugate_matrix([x | _] = m) when not is_list(x) do
    Enum.map(m, &Complex.conjugate/1)
  end

  defp conjugate_matrix(m) do
    Enum.map(m, fn row -> Enum.map(row, &Complex.conjugate/1) end)
  end

  defp transpose_matrix([x | _] = m) when not is_list(x) do
    Enum.map(m, &[&1])
  end

  defp transpose_matrix(m) do
    Enum.zip_with(m, & &1)
  end

  defp reverse_matrix([x | _] = m) when not is_list(x) do
    Enum.reverse(m)
  end

  defp reverse_matrix(m) do
    m
    |> Enum.map(&Enum.reverse/1)
    |> Enum.reverse()
  end

  defp matrix_to_binary([r | _] = m, type) when is_list(r) do
    match_types [type] do
      for row <- m, number <- row, into: "", do: <<write!(number, 0)>>
    end
  end

  defp matrix_to_binary(list, type) when is_list(list) do
    match_types [type] do
      for number <- list, into: "", do: <<write!(number, 0)>>
    end
  end

  defp binary_to_vector(bin, type) do
    match_types [type] do
      for <<match!(x, 0) <- bin>>, into: [], do: read!(x, 0)
    end
  end

  defp binary_to_matrix(bin, type, {_, num_cols}) do
    bin
    |> binary_to_vector(type)
    |> Enum.chunk_every(num_cols)
  end

  defp get_matrix_column(m, col) do
    Enum.map(m, fn row ->
      Enum.at(row, col)
    end)
  end
end
