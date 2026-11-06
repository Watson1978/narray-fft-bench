# frozen_string_literal: true

module FFTBench
  module Problems
    # E. Registers an image against shifted copies of itself by phase
    # correlation. J = roll(I, s) must give back s exactly, through fft2 and
    # through rfft2. The fractional copies (I with every bin's phase turned)
    # are read to a fraction of a pixel by the three-point sinc estimate
    # d = r1 / (r1 + r0) (Foroosh et al. 2002), which fits the sinc-shaped
    # peak of phase correlation where a parabola does not.
    class PhaseCorrelation < Problem
      register :phase_correlation

      VARIANTS = %w[fft2 rfft2].freeze
      DIFFER_TOL = { 'fp64' => 2.2e-15, 'fp32' => 2.4e-6 }.freeze
      PEAK_TOL = { 'fp64' => 1e-10, 'fp32' => 1.2e-6 }.freeze
      FRACTION_TOL = 0.05

      def run(fixture)
        images = { 'even' => fixture.device(:image) }
        images['odd'] = fixture.device(:image_odd) if fixture.arrays.key?('image_odd')
        shifts = fixture[:shifts].to_a.map { |row| row.map(&:to_i) }
        integer = images.transform_values { |image| shifts.map { |s| locate(image, roll(image, s), s) } }
        image = images['even']
        fractional = fixture[:frac_shifts].to_a.each_with_index.map do |truth, k|
          estimate(surface(image, fixture.device(:frac_images)[k, true, true], 'fft2'), truth)
        end
        round_trips = images.transform_values do |x|
          [Gates.mismatches(XFT.fftshift(XFT.ifftshift(x)), x), Gates.mismatches(XFT.ifftshift(XFT.fftshift(x)), x)].sum
        end
        { 'integer' => integer, 'fractional' => fractional, 'round_trips' => round_trips,
          'values' => values(integer['even'], fractional) }
      end

      def gate(result)
        gates = result['integer'].flat_map do |name, found|
          [Gates.eq("#{name}.wrong_shifts", found.sum { |f| VARIANTS.count { |v| f[v][:shift] != f[:truth] } }, 0),
           Gates.le("#{name}.paths_differ", found.map { |f| f[:differ] }.max, DIFFER_TOL.fetch(FFTBench.dtype_name)),
           Gates.ge("#{name}.peak_min", found.map { |f| f['fft2'][:peak] }.min, 1 - PEAK_TOL.fetch(FFTBench.dtype_name))]
        end
        gates << Gates.le('fractional.error', result['fractional'].map { |f| f[:error] }.max, FRACTION_TOL)
        gates + result['round_trips'].map { |name, count| Gates.eq("#{name}.shift_round_trip", count, 0) }
      end

      def bench_step(fixture, variant = nil)
        path = variant || VARIANTS.first
        raise Error, "phase_correlation has no variant #{path}; known: #{VARIANTS.join(', ')}" unless VARIANTS.include?(path)

        image = fixture.device(:image)
        shifted = roll(image, fixture[:shifts].to_a.first.map(&:to_i))
        -> { surface(image, shifted, path).max_index }
      end

      private

      def surface(i, j, path)
        if path == 'rfft2'
          cross = XFT.rfft2(i) * XFT.rfft2(j).conj
          XFT.fftshift(XFT.irfft2(cross / cross.abs, s: i.shape))
        else
          cross = XFT.fft2(i) * XFT.fft2(j).conj
          XFT.fftshift(XFT.ifft2(cross / cross.abs).real)
        end
      end

      def locate(i, j, truth)
        surfaces = VARIANTS.to_h { |path| [path, surface(i, j, path)] }
        found = surfaces.transform_values do |r|
          p = FFTBench.scalar(r.max_index).divmod(r.shape[1])
          { shift: p.zip(r.shape).map { |pk, n| (n / 2) - pk }, peak: FFTBench.scalar(r.max) }
        end
        found.merge(truth: truth, differ: Gates.rel_max(surfaces['rfft2'], surfaces['fft2']))
      end

      # The peak is never on the border: every shift is under n / 4.
      def estimate(r, truth)
        p0, p1 = FFTBench.scalar(r.max_index).divmod(r.shape[1])
        column = FFTBench.host(r[(p0 - 1)..(p0 + 1), p1]).to_a
        row = FFTBench.host(r[p0, (p1 - 1)..(p1 + 1)]).to_a
        found = [[p0, column], [p1, row]].zip(r.shape).map { |(p, (left, top, right)), n| (n / 2) - p - offset(left, top, right) }
        { shift: found, error: found.zip(truth).map { |f, t| (f - t).abs }.max }
      end

      def offset(left, top, right)
        right > left ? right / (right + top) : -left / (left + top)
      end

      def roll(x, shift)
        blocks = shift.zip(x.shape).map do |s, n|
          s %= n
          [[0...(n - s), s...n], [(n - s)...n, 0...s]].reject { |source, _| source.size.zero? }
        end
        y = x.class.zeros(*x.shape)
        blocks[0].product(blocks[1]).each { |(r_from, r_to), (c_from, c_to)| y[r_to, c_to] = x[r_from, c_from] }
        y
      end

      def values(integer, fractional)
        estimates = fractional.each_with_index.flat_map do |f, k|
          f[:shift].each_with_index.map { |s, axis| ["fractional.#{k}.axis#{axis}", s] }
        end
        { 'shifts_found' => integer.count { |f| f['fft2'][:shift] == f[:truth] },
          'peak_min' => integer.map { |f| f['fft2'][:peak] }.min }.merge(estimates.to_h)
      end
    end
  end
end
