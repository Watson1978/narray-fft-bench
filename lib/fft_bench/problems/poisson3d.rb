# frozen_string_literal: true

module FFTBench
  module Problems
    # C. The potential of a Gaussian charge in a periodic box, nabla^2 phi =
    # -rho, by phi_hat = rho_hat / k^2 with the k = 0 mode dropped. The
    # differences of phi between grid points under L / 4 must match the
    # periodic solution from Ewald summation in the fixture. The complex
    # path, the Laplacian and the round trips must agree too.
    class Poisson3d < Problem
      register :poisson3d

      VARIANTS = %w[rfftn fftn axes01].freeze
      REFERENCE_TOL = 1e-6
      LAPLACIAN_TOL = { 'fp64' => 4e-13, 'fp32' => 2.9e-4 }.freeze
      TOL = { 'fp64' => 2.8e-15, 'fp32' => 3.2e-6 }.freeze

      def run(fixture)
        spec = fixture.attr(:spec)
        rho = fixture.device(:rho)
        n = rho.shape[0]
        g = grid(n, spec['box'])
        phi = solve(rho, g, 'rfftn')
        found = potentials(phi, fixture[:points].to_a)
        reference = fixture[:reference].to_a
        odd = fixture.device(:rho_odd)
        { 'reference' => difference_error(found, reference),
          'complex' => Gates.rel_max(solve(rho, g, 'fftn'), phi),
          'laplacian' => Gates.rel_max(XFT.irfftn(XFT.rfftn(phi) * g[:minus_k2], s: rho.shape), -(rho - rho.mean)),
          'round_trip' => Gates.rel_max(XFT.irfftn(XFT.rfftn(rho), s: rho.shape), rho),
          'odd_round_trip' => Gates.rel_max(XFT.irfftn(XFT.rfftn(odd), s: odd.shape), odd),
          'axes01' => Gates.rel_max(XFT.fft(XFT.rfftn(rho, axes: [0, 1]), axis: 2), XFT.rfftn(rho, axes: [2, 0, 1])),
          'values' => { 'phi_span' => found.max - found.min, 'phi_axis_8' => found[2] - found[0],
                        'odd_length_without_s' => XFT.irfftn(XFT.rfftn(odd)).shape[-1] } }
      end

      def gate(result)
        dtype = FFTBench.dtype_name
        [Gates.le('reference', result['reference'], REFERENCE_TOL),
         Gates.le('laplacian', result['laplacian'], LAPLACIAN_TOL.fetch(dtype))] +
          %w[complex round_trip odd_round_trip axes01].map { |name| Gates.le(name, result[name], TOL.fetch(dtype)) }
      end

      def bench_step(fixture, variant = nil)
        path = variant || VARIANTS.first
        raise Error, "poisson3d has no variant #{path}; known: #{VARIANTS.join(', ')}" unless VARIANTS.include?(path)

        rho = fixture.device(:rho)
        g = grid(rho.shape[0], fixture.attr(:spec)['box'])
        -> { solve(rho, g, path) }
      end

      private

      # Angular wavenumbers 2 pi m / L; the last axis is one-sided for rfftn.
      def grid(n, box)
        k = XF.cast(XFT.fftfreq(n, d: box / n) * (2 * Math::PI))
        half = XF.cast(XFT.rfftfreq(n, d: box / n) * (2 * Math::PI))
        k0 = k.reshape(n, 1, 1)
        k1 = k.reshape(1, n, 1)
        both = (k0 * k0) + (k1 * k1)
        half2 = both + (half * half).reshape(1, 1, half.size)
        full2 = both + (k * k).reshape(1, 1, n)
        { inverse_half: inverse(half2), inverse_full: inverse(full2), minus_k2: -half2, n: n,
          inverse_slice: inverse((k0 * k0) + (half * half).reshape(1, half.size, 1)) }
      end

      def inverse(k2)
        safe = k2.dup
        safe[0, 0, 0] = 1
        out = 1 / safe
        out[0, 0, 0] = 0
        out
      end

      def solve(rho, g, path)
        case path
        when 'rfftn' then XFT.irfftn(XFT.rfftn(rho) * g[:inverse_half], s: rho.shape)
        when 'fftn' then XFT.ifftn(XFT.fftn(rho) * g[:inverse_full]).real
        else XFT.irfftn(XFT.rfftn(rho, axes: [0, 1]) * g[:inverse_slice], s: rho.shape[0, 2], axes: [0, 1])
        end
      end

      # The charge sits at index n / 2 on every axis.
      def potentials(phi, points)
        c = phi.shape[0] / 2
        points.map { |p| FFTBench.scalar(phi[c + p[0].to_i, c + p[1].to_i, c + p[2].to_i]) }
      end

      # Differences from the first point, so the constant of each drops out.
      def difference_error(found, reference)
        df = found.map { |v| v - found[0] }
        dr = reference.map { |v| v - reference[0] }
        df.zip(dr).map { |a, b| (a - b).abs }.max / dr.map(&:abs).max
      end
    end
  end
end
