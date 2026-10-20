defmodule Nx.Floating do
  @moduledoc """
  Functions for loading and dumping floating-point formats
  not supported by the Erlang VM.
  """

  import Bitwise

  @compile {:inline, round_to_nearest_even: 4, shift_to_nearest_even: 2}

  ## BF16

  @doc """
  Loads a BF16 (Brain Float 16) value from its binary representation.

  Returns a float, `:nan`, `:infinity`, or `:neg_infinity`.

  ## Examples

      iex> Nx.Floating.load_bf16(Nx.Floating.dump_bf16(1.0))
      1.0

      iex> Nx.Floating.load_bf16(Nx.Floating.dump_bf16(:infinity))
      :infinity

      iex> Nx.Floating.load_bf16(Nx.Floating.dump_bf16(:neg_infinity))
      :neg_infinity

  """
  def load_bf16(<<0xFF80::16-native>>), do: :neg_infinity
  def load_bf16(<<0x7F80::16-native>>), do: :infinity

  if System.endianness() == :little do
    def load_bf16(<<1::1, _::7, _sign::1, 127::7>>), do: :nan

    def load_bf16(bf16) do
      <<x::float-little-32>> = <<0::16, bf16::binary>>
      x
    end
  else
    def load_bf16(<<_sign::1, 255::8, _::7>>), do: :nan

    def load_bf16(bf16) do
      <<x::float-big-32>> = <<bf16::binary, 0::16>>
      x
    end
  end

  @doc """
  Dumps an Elixir value to BF16 (Brain Float 16) binary representation.

  ## Examples

      iex> Nx.Floating.load_bf16(Nx.Floating.dump_bf16(1.0))
      1.0

      iex> Nx.Floating.load_bf16(Nx.Floating.dump_bf16(-2.5))
      -2.5

      iex> Nx.Floating.load_bf16(Nx.Floating.dump_bf16(:nan))
      :nan

  """
  def dump_bf16(x) when is_number(x) do
    <<sign::1, exponent::11, fraction::52>> = <<x::float-64>>
    bits = round_to_nearest_even(exponent, fraction, 8, 7)
    <<sign <<< 15 ||| min(bits, 0x7F80)::16-native>>
  end

  def dump_bf16(:infinity), do: unquote(Nx.Type.infinity_binary({:bf, 16}))
  def dump_bf16(:neg_infinity), do: unquote(Nx.Type.neg_infinity_binary({:bf, 16}))
  def dump_bf16(:nan), do: unquote(Nx.Type.nan_binary({:bf, 16}))

  ## F8 (E5M2)

  @doc """
  Loads an F8 (E5M2) value from its binary representation.

  Returns a float, `:nan`, `:infinity`, or `:neg_infinity`.

  ## Examples

      iex> Nx.Floating.load_f8(Nx.Floating.dump_f8(1.0))
      1.0

      iex> Nx.Floating.load_f8(Nx.Floating.dump_f8(:infinity))
      :infinity

      iex> Nx.Floating.load_f8(Nx.Floating.dump_f8(:neg_infinity))
      :neg_infinity

  """
  def load_f8(<<0xFC::8-native>>), do: :neg_infinity
  def load_f8(<<0x7C::8-native>>), do: :infinity
  def load_f8(<<_sign::1, 31::5, mantissa::2>>) when mantissa != 0, do: :nan

  # E5M2 is the high byte of an F16.
  def load_f8(<<byte::8>>) do
    <<float::float-big-16>> = <<byte, 0>>
    float
  end

  @doc """
  Dumps an Elixir value to F8 (E5M2) binary representation.

  ## Examples

      iex> Nx.Floating.load_f8(Nx.Floating.dump_f8(1.0))
      1.0

      iex> Nx.Floating.load_f8(Nx.Floating.dump_f8(-2.0))
      -2.0

      iex> Nx.Floating.load_f8(Nx.Floating.dump_f8(:nan))
      :nan

  """
  def dump_f8(x) when is_number(x) do
    <<sign::1, exponent::11, fraction::52>> = <<x::float-64>>
    bits = round_to_nearest_even(exponent, fraction, 5, 2)
    <<sign::1, min(bits, 0x7C)::7>>
  end

  def dump_f8(:infinity), do: unquote(Nx.Type.infinity_binary({:f, 8}))
  def dump_f8(:neg_infinity), do: unquote(Nx.Type.neg_infinity_binary({:f, 8}))
  def dump_f8(:nan), do: unquote(Nx.Type.nan_binary({:f, 8}))

  ## F8 E4M3FN

  @doc """
  Loads an F8 E4M3FN value from its binary representation.

  Returns a float or `:nan` (E4M3FN has no infinity).

  ## Examples

      iex> Nx.Floating.load_f8_e4m3fn(Nx.Floating.dump_f8_e4m3fn(1.0))
      1.0

      iex> Nx.Floating.load_f8_e4m3fn(Nx.Floating.dump_f8_e4m3fn(:nan))
      :nan

      iex> Nx.Floating.load_f8_e4m3fn(Nx.Floating.dump_f8_e4m3fn(0.0))
      0.0

      iex> Nx.Floating.load_f8_e4m3fn(Nx.Floating.dump_f8_e4m3fn(448.0))
      448.0

  """
  # E4M3FN format: 1 sign bit, 4 exponent bits, 3 mantissa bits
  # Exponent bias: 7
  # Per OFP8 spec: E4M3FN has NO infinity (FN = "Finite, No infinities")
  # Only S.1111.111 is NaN; all other S.1111.xxx are finite values

  # Only mantissa = 111 (0x7) is NaN in E4M3FN
  def load_f8_e4m3fn(<<_sign::1, 15::4, 7::3>>), do: :nan

  def load_f8_e4m3fn(<<0::1, 0::4, mantissa::3>>) do
    # Denormalized positive
    :math.pow(2, -6) * (mantissa / 8.0)
  end

  def load_f8_e4m3fn(<<1::1, 0::4, mantissa::3>>) do
    # Denormalized negative
    -:math.pow(2, -6) * (mantissa / 8.0)
  end

  def load_f8_e4m3fn(<<sign::1, exp::4, mantissa::3>>) do
    # Normalized
    float = :math.pow(2, exp - 7) * (1 + mantissa / 8.0)

    case sign do
      0 -> float
      _ -> -float
    end
  end

  @doc """
  Dumps an Elixir value to F8 E4M3FN binary representation.

  Finite values are clamped to the E4M3FN range [-448.0, 448.0].
  Since E4M3FN has no infinity, `:infinity` and `:neg_infinity` saturate
  to max/min finite values. Only `:nan` maps to NaN.

  ## Examples

      iex> Nx.Floating.dump_f8_e4m3fn(1.0)
      <<0x38>>

      iex> Nx.Floating.dump_f8_e4m3fn(0.0)
      <<0x00>>

      iex> Nx.Floating.dump_f8_e4m3fn(448.0)
      <<0x7E>>

      iex> Nx.Floating.dump_f8_e4m3fn(-448.0)
      <<0xFE>>

      iex> Nx.Floating.dump_f8_e4m3fn(:infinity)
      <<0x7E>>

      iex> Nx.Floating.dump_f8_e4m3fn(:neg_infinity)
      <<0xFE>>

      iex> Nx.Floating.dump_f8_e4m3fn(:nan)
      <<0x7F>>

  """
  # E4M3FN: 1 sign, 4 exponent (bias 7), 3 mantissa
  # Max value: 448.0 (0x7E), Min value: -448.0 (0xFE)
  def dump_f8_e4m3fn(x) when is_number(x) do
    <<sign::1, exponent::11, fraction::52>> = <<x::float-64>>
    bits = round_to_nearest_even(exponent, fraction, 4, 3)
    <<sign::1, min(bits, 0x7E)::7>>
  end

  # Saturate infinity to max/min finite values (not NaN)
  # This preserves sign and matches overflow saturation behavior
  def dump_f8_e4m3fn(:infinity), do: unquote(Nx.Type.infinity_binary({:f8_e4m3fn, 8}))
  def dump_f8_e4m3fn(:neg_infinity), do: unquote(Nx.Type.neg_infinity_binary({:f8_e4m3fn, 8}))
  def dump_f8_e4m3fn(:nan), do: unquote(Nx.Type.nan_binary({:f8_e4m3fn, 8}))

  # The exponent and mantissa bits of an f64, given by its exponent and
  # fraction fields, rounded to the nearest value of the format, ties to even.
  # On overflow the bits pass the largest exponent.
  defp round_to_nearest_even(exponent, fraction, exponent_bits, mantissa_bits) do
    bias = (1 <<< (exponent_bits - 1)) - 1
    unbiased = exponent - 1023
    dropped = 52 - mantissa_bits

    cond do
      exponent == 0 ->
        0

      unbiased >= 1 - bias ->
        ((unbiased + bias) <<< mantissa_bits) + shift_to_nearest_even(fraction, dropped)

      true ->
        shift_to_nearest_even(1 <<< 52 ||| fraction, dropped + 1 - bias - unbiased)
    end
  end

  defp shift_to_nearest_even(value, shift) do
    quotient = value >>> shift
    remainder = value &&& (1 <<< shift) - 1
    half = 1 <<< (shift - 1)

    if remainder > half or (remainder == half and (quotient &&& 1) == 1) do
      quotient + 1
    else
      quotient
    end
  end
end
