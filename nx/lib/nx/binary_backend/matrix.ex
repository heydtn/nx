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
    y = (b - dot_matrix(row, acc)) / value
    do_ts(rows, bs, idx + 1, acc ++ [y])
  end

  defp do_ts([], [], _idx, acc), do: acc

  # LAPACK-based solvers, such as PyTorch's, report a matrix as singular when
  # the LU factorization finds an exact zero pivot. Which matrices do depends
  # on the arithmetic, so this repeats LAPACK's as PyTorch runs it: real
  # matrices are factored transposed, as PyTorch does for row-major tensors,
  # the pivot row is swapped into place, multipliers use the reciprocal of the
  # pivot unless it is below the smallest normal number, and updates are
  # fused multiply-adds, rounded once.
  def lu_zero_pivot?(a_data, a_type, a_shape, type) do
    component_type =
      case type do
        {:c, size} -> {:f, div(size, 2)}
        _ -> type
      end

    matrix = binary_to_matrix(a_data, a_type, a_shape)

    rounded =
      if a_type == type,
        do: matrix,
        else: Enum.map(matrix, fn row -> Enum.map(row, &round_to_type(&1, component_type)) end)

    rows = if match?({:c, _}, type), do: rounded, else: transpose_matrix(rounded)
    eliminate(rows, component_type, smallest_normal(component_type))
  end

  defp eliminate([], _type, _smallest_normal), do: false

  defp eliminate(rows, type, smallest_normal) do
    [[pivot | pivot_row] | rest] = swap_pivot_row(rows)

    cond do
      zero?(pivot) ->
        true

      modulus(pivot) >= smallest_normal ->
        reciprocal = reciprocal(pivot, type)

        rest
        |> Enum.map(fn [x | row] ->
          subtract_row(row, pivot_row, multiply(x, reciprocal, type), type)
        end)
        |> eliminate(type, smallest_normal)

      true ->
        rest
        |> Enum.map(fn [x | row] ->
          subtract_row(row, pivot_row, round_to_type(x / pivot, type), type)
        end)
        |> eliminate(type, smallest_normal)
    end
  end

  # Like LAPACK's i?amax: the first row whose leading entry has the largest
  # magnitude, swapped with the first row. Complex magnitudes are |re| + |im|,
  # and NaN is never larger.
  defp swap_pivot_row([[first | _] = first_row | rest] = rows) do
    {_, index} =
      rest
      |> Enum.with_index(1)
      |> Enum.reduce({pivot_magnitude(first), 0}, fn {[x | _], i}, {max, max_index} ->
        magnitude = pivot_magnitude(x)
        if greater?(magnitude, max), do: {magnitude, i}, else: {max, max_index}
      end)

    if index == 0,
      do: rows,
      else: [Enum.at(rows, index) | List.replace_at(rest, index - 1, first_row)]
  end

  defp zero?(%Complex{re: re, im: im}), do: re == 0 and im == 0
  defp zero?(x), do: x == 0

  defp modulus(%Complex{} = z), do: Complex.abs(z)
  defp modulus(x), do: Complex.abs(x)

  defp pivot_magnitude(%Complex{re: re, im: im}), do: Complex.abs(re) + Complex.abs(im)
  defp pivot_magnitude(x), do: Complex.abs(x)

  defp greater?(:nan, _), do: false
  defp greater?(_, :nan), do: false
  defp greater?(a, b), do: a > b

  # Complex 1 / z uses Smith's algorithm in f64, as LAPACK's complex
  # division does, and is then rounded to the component type
  defp reciprocal(%Complex{re: re, im: im}, type) when is_float(re) and is_float(im) do
    if abs(re) <= abs(im) do
      ratio = re / im
      denominator = im * fma(ratio, ratio, 1.0, {:f, 64})

      Complex.new(
        round_to_type(ratio / denominator, type),
        round_to_type(-1.0 / denominator, type)
      )
    else
      ratio = im / re
      denominator = re * fma(ratio, ratio, 1.0, {:f, 64})

      Complex.new(
        round_to_type(1.0 / denominator, type),
        round_to_type(-ratio / denominator, type)
      )
    end
  end

  defp reciprocal(x, type), do: round_to_type(1 / x, type)

  defp multiply(%Complex{re: x_re, im: x_im}, %Complex{re: q_re, im: q_im}, type)
       when is_float(x_re) and is_float(x_im) and is_float(q_re) and is_float(q_im) do
    Complex.new(
      fma(x_re, q_re, round_to_type(-x_im * q_im, type), type),
      fma(x_im, q_re, round_to_type(x_re * q_im, type), type)
    )
  end

  defp multiply(x, q, type), do: round_to_type(x * q, type)

  defp subtract_row([x | xs], [u | us], multiplier, type),
    do: [subtract_product(x, multiplier, u, type) | subtract_row(xs, us, multiplier, type)]

  defp subtract_row([], [], _multiplier, _type), do: []

  defp subtract_product(x, l, u, type) when is_float(x) and is_float(l) and is_float(u),
    do: fma(-l, u, x, type)

  # In the order Apple's complex kernels use
  defp subtract_product(
         %Complex{re: x_re, im: x_im},
         %Complex{re: l_re, im: l_im},
         %Complex{re: u_re, im: u_im},
         type
       )
       when is_float(x_re) and is_float(x_im) and is_float(l_re) and is_float(l_im) and
              is_float(u_re) and is_float(u_im) do
    Complex.new(
      fma(l_im, u_im, fma(-l_re, u_re, x_re, type), type),
      fma(-l_re, u_im, fma(-l_im, u_re, x_im, type), type)
    )
  end

  defp subtract_product(x, l, u, type), do: round_to_type(x - l * u, type)

  # a * b + c rounded once, as a fused multiply-add computes it.
  #
  # The f64 version follows Boldo and Melquiond: the rounding errors of the
  # product and of the sum are recovered exactly, and their sum is rounded to
  # odd before the final addition, which then rounds the exact result. Values
  # that could overflow or underflow those steps use integer arithmetic.
  # The exact-product and exact-sum shortcuts matter for matrices with
  # integer or zero entries, which is where singular matrices come from.
  @veltkamp 134_217_729.0

  defp fma(a, b, c, {:f, 64}) when is_float(a) and is_float(b) and is_float(c) do
    product = a * b

    cond do
      a == 0.0 or b == 0.0 or c == 0.0 ->
        c + product

      abs(a) < 1.0e-140 or abs(a) > 1.0e140 or abs(b) < 1.0e-140 or abs(b) > 1.0e140 or
          abs(c) > 1.0e300 ->
        exact_fma(a, b, c)

      true ->
        a_split = @veltkamp * a
        a_high = a_split - (a_split - a)
        a_low = a - a_high
        b_split = @veltkamp * b
        b_high = b_split - (b_split - b)
        b_low = b - b_high

        product_error =
          a_high * b_high - product + a_high * b_low + a_low * b_high + a_low * b_low

        if product_error == 0.0 do
          c + product
        else
          sum = c + product
          product_part = sum - c
          sum_error = c - (sum - product_part) + (product - product_part)

          if sum_error == 0.0,
            do: sum + product_error,
            else: sum + round_to_odd(sum_error, product_error)
        end
    end
  end

  # The product of two f32 values is exact in f64, so the f64 sum is the exact
  # result rounded once. Rounding it again to f32 is only wrong when it lies
  # exactly halfway between two f32 values; then, or outside the f32 normal
  # range, the sum is recomputed rounded to odd, which makes the final
  # rounding exact.
  defp fma(a, b, c, {:f, 32}) when is_float(a) and is_float(b) and is_float(c) do
    sum = a * b + c

    cond do
      sum == 0.0 ->
        sum

      abs(sum) < 1.1754943508222875e-38 or abs(sum) > 3.4028234663852886e38 or
          match?(<<_::35, 1::1, 0::28>>, <<sum::float>>) ->
        round_to_type(round_to_odd(a * b, c), {:f, 32})

      true ->
        round_to_type(sum, {:f, 32})
    end
  end

  defp fma(a, b, c, type) when is_float(a) and is_float(b) and is_float(c),
    do: round_to_type(round_to_odd(a * b, c), type)

  defp fma(a, b, c, type), do: round_to_type(a * b + c, type)

  # a + b, with the last bit forced to 1 when the sum is inexact
  defp round_to_odd(a, b) do
    sum = a + b
    b_part = sum - a
    error = a - (sum - b_part) + (b - b_part)

    if error == 0.0 do
      sum
    else
      <<sign::1, magnitude::63>> = <<sum::float>>

      cond do
        Bitwise.band(magnitude, 1) == 1 ->
          sum

        error > 0 == (sign == 0) ->
          <<odd::float>> = <<sign::1, magnitude + 1::63>>
          odd

        true ->
          <<odd::float>> = <<sign::1, magnitude - 1::63>>
          odd
      end
    end
  end

  defp exact_fma(a, b, c) do
    {a_mantissa, a_exponent} = float_to_dyadic(a)
    {b_mantissa, b_exponent} = float_to_dyadic(b)
    {c_mantissa, c_exponent} = float_to_dyadic(c)
    product_exponent = a_exponent + b_exponent
    exponent = min(c_exponent, product_exponent)

    mantissa =
      Bitwise.bsl(a_mantissa * b_mantissa, product_exponent - exponent) +
        Bitwise.bsl(c_mantissa, c_exponent - exponent)

    dyadic_to_float(mantissa, exponent)
  end

  # Splits a float into an integer mantissa and exponent, x = mantissa * 2 ** exponent
  defp float_to_dyadic(x) do
    <<sign::1, exponent::11, fraction::52>> = <<x::float-64>>

    {mantissa, exponent} =
      if exponent == 0,
        do: {fraction, -1074},
        else: {fraction + Bitwise.bsl(1, 52), exponent - 1075}

    if sign == 1, do: {-mantissa, exponent}, else: {mantissa, exponent}
  end

  # Rounds mantissa * 2 ** exponent to the nearest f64, ties to even
  defp dyadic_to_float(0, _exponent), do: 0.0

  defp dyadic_to_float(mantissa, exponent) do
    magnitude = abs(mantissa)
    shift = max(bit_length(magnitude) - 53, -1074 - exponent)

    {rounded, rounded_exponent} =
      if shift > 0 do
        quotient = Bitwise.bsr(magnitude, shift)
        remainder = magnitude - Bitwise.bsl(quotient, shift)
        half = Bitwise.bsl(1, shift - 1)
        round_up? = remainder > half or (remainder == half and rem(quotient, 2) == 1)
        {if(round_up?, do: quotient + 1, else: quotient), exponent + shift}
      else
        {magnitude, exponent}
      end

    cond do
      rounded_exponent + bit_length(rounded) > 1024 and mantissa > 0 -> :infinity
      rounded_exponent + bit_length(rounded) > 1024 -> :neg_infinity
      mantissa > 0 -> rounded * :math.pow(2, rounded_exponent)
      true -> -rounded * :math.pow(2, rounded_exponent)
    end
  end

  defp bit_length(n), do: n |> Integer.to_string(2) |> byte_size()

  defp smallest_normal({:f, 64}), do: 2.2250738585072014e-308
  defp smallest_normal({:f, 32}), do: 1.1754943508222875e-38

  defp smallest_normal(type) do
    type
    |> Nx.Constants.smallest_positive_normal(backend: Nx.BinaryBackend)
    |> Nx.to_number()
  end

  defp round_to_type(%Complex{re: re, im: im}, type),
    do: Complex.new(round_to_type(re, type), round_to_type(im, type))

  defp round_to_type(number, _type) when is_atom(number), do: number
  defp round_to_type(number, type) when is_integer(number), do: round_to_type(number * 1.0, type)
  defp round_to_type(number, {:f, 64}), do: number

  defp round_to_type(number, {:f, 32}) do
    case <<number::float-32>> do
      <<rounded::float-32>> -> rounded
      _ when number > 0 -> :infinity
      _ -> :neg_infinity
    end
  end

  defp round_to_type(number, type), do: round_float_to_type(number, type)

  defp round_float_to_type(number, type) do
    match_types [type] do
      <<match!(rounded, 0)>> = <<write!(number, 0)>>
      read!(rounded, 0)
    end
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
