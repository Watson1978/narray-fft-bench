# narray-fft-bench 実装プラン

作成日: 2026-10-17
目的: `Cumo::FFT`(cuFFT 実装、別プラン `cumo-fft-plan.md`)を **numo-pocketfft と互換か(計算結果を含む)**、**CuPy と比べて速いか** の 2 点で検証する。narray-llm が LLM で、narray-linalg-bench が線形代数で果たした役割を、FFT で果たすプロジェクト。

進め方は narray-linalg-bench の作法をそのまま継ぐ。**前の段階の受け入れテストが通るまで次の段階のコードを書かない。** 関門(正しさ)と計測(速さ)は別のコマンドに分ける。

---

## 0. 結論(先に読むところ)

- **A〜G の 7 題材**で、`Cumo::FFT` の 18 関数(numpy.fft の 18)と `Cumo::Pocketfft` の 13 関数をすべて踏む(§3 のカバー表)。関門は物理的・解析的に決まる性質で、**A・D・E・F は完全一致**(トークン列、整数の桁、シフト量、ビット)を要求できるので許容誤差の議論が要らない。
- **4 実装を並べる**: Ruby(`XM` の切り替えで Numo / Cumo)、Python(numpy+scipy / CuPy)。**同じ題材を同じアルゴリズムで、同じ fixture を読んで**動かす。
- **Numo 側の FFT は numo-pocketfft で、numpy 流の API ではない**(倍精度のみ、`n:` / `axis:` / `norm:` なし、1 次元用と 2 次元用で関数が分かれる)。題材コードは numpy 流で書き、Numo では `lib/fft_bench/numo_fft.rb` の薄いシム(swapaxes・ゼロ埋め・正規化を Ruby で足す)を通す。シムは **cumo 側の `Cumo::Pocketfft` 互換層の鏡像**で、互換層の試験にもなる。
- **CuPy には FFT の関数がすべてある**(`cupy.fft` 18 関数 + `cupyx.scipy.signal.fftconvolve`)。Linalg のような埋め合わせは要らない。差が出るのは cuFFT そのものではなく、**プランのキャッシュ、前後のコピー(軸の入れ替え、ゼロ埋め、C2R の退避)、正規化の 1 パス**というラッパー側。
- dtype は **関門は DFloat、計測は SFloat と DFloat の両方**。Numo(pocketfft)は倍精度しか無いので CPU 表は DFloat だけ。GPU 表は SFloat が主。
- fixture は **Numo で生成して `data/` にバイナリで置き、4 実装が同じバイトを読む**。乱数は Numo で seed 固定、1 か所だけ。
- **計測は「初回(プラン生成込み)」と「定常(キャッシュ命中)」を分けて出す。** FFT はここが Linalg と違う。

---

## 1. リポジトリの形

```
narray-fft-bench/
├── AGENTS.md                 # 作法(narray-linalg-bench のものを FFT 向けに書き直す)
├── README.md                 # 結果表(GPU: Cumo / CuPy、CPU: Numo / numpy+scipy)
├── Gemfile                   # numo-narray-alt, numo-pocketfft, numo-linalg-alt(dot 用), test-unit; group :gpu { cumo }
├── Rakefile                  # rake test / rake gate / rake bench
├── lib/
│   ├── fft_bench.rb
│   └── fft_bench/
│       ├── backend.rb        # XM / XFT / XF、GPU=1、DTYPE=、scalar 読み戻し
│       ├── numo_fft.rb       # Numo 用シム: numpy 流 API を Numo::Pocketfft の上に(§1.1)
│       ├── fixtures.rb       # data/ の読み書き(to_binary / from_binary)
│       ├── gates.rb          # 相対誤差、Parseval、完全一致の共通関数
│       ├── timer.rb          # best-of-N、ウォームアップ、初回/定常の分離、区間末の同期
│       └── problems/
│           ├── whisper_mel.rb         # A
│           ├── taylor_green.rb        # B
│           ├── poisson3d.rb           # C
│           ├── bignum_mul.rb          # D
│           ├── phase_correlation.rb   # E
│           ├── ofdm.rb                # F
│           └── spectrum.rb            # G
├── script/
│   ├── make_fixtures.rb      # data/ を生成(Numo、DFloat、決定的)
│   ├── gate.rb               # 関門だけを全題材で回す(GPU=1 で Cumo)
│   └── bench.rb              # 1 題材 × 1 サイズ × 1 dtype × 初回/定常 を計測して 1 行出す
├── python/
│   ├── requirements.txt      # numpy, scipy, cupy-cuda13x
│   ├── backend.py            # xp / xfft / xsig の切り替え(GPU=1 で cupy)、同期、プランキャッシュの統計
│   ├── fixtures.py
│   ├── gates.py
│   ├── problems/             # Ruby と 1 対 1 の 7 ファイル
│   ├── gate.py
│   └── bench.py
├── bench/
│   ├── run.sh                # narray-llm の run.sh を継ぐ(インターリーブ、対照系列)
│   ├── conditions.tsv        # 題材 × サイズ × dtype × 初回/定常 × 実装コマンド
│   └── aggregate.py          # 比の中央値、n/10
├── data/                     # fixture(git には入れない。make_fixtures.rb で再生成)
├── test/                     # 受け入れテスト(両バックエンドで通す、GPU 無しでも Numo で全部通す)
│   ├── parity_test.rb        # Cumo::FFT vs Numo シム(numpy 流 API どうし)
│   └── pocketfft_parity_test.rb  # Cumo::Pocketfft vs Numo::Pocketfft(dtype・例外・fftconvolve の端)
└── docs/
    ├── plans/PLAN-fft-bench.md      # このファイル
    ├── method.md                    # 比較の条件と手順(初回/定常、プランキャッシュ、同期)
    ├── cupy-notes.md                # CuPy の実装との違い(埋め合わせは無いが、挙動差を記録)
    ├── cumo-issues.md               # 見つけた cumo の問題、棄却した仮説
    └── results/<題材>.md            # 題材ごとの数字と経過
```

### 1.1 backend.rb と Numo シムの要点

```ruby
require 'numo/narray'
require 'numo/pocketfft'       # 無ければ LoadError で止める(CPU 側の前提)
begin; require 'numo/linalg'; rescue LoadError; end   # dot を BLAS に(無くても動く)
HM = Numo

if ENV['GPU'].to_s =~ /\A(1|on|true)\z/i
  require 'cumo/narray'
  require 'cumo/fft'
  XM  = Cumo
  XFT = Cumo::FFT               # numpy 流 API をそのまま
else
  XM  = Numo
  XFT = FFTBench::NumoFFT       # numpy 流 API を Numo::Pocketfft の上に組んだシム
end
XF = ENV['DTYPE'] == 'fp64' ? XM::DFloat : XM::SFloat   # 既定 SFloat。関門は fp64
```

- 題材コードは `Numo::` / `Cumo::` / `Pocketfft` を直接書かない。`XM`、`XFT`、`XF` だけ。呼び方は `XFT.fft(a, n: nil, axis: -1, norm: nil)`、`XFT.fftn(a, s: nil, axes: nil, norm: nil)`、`XFT.fftconvolve(a, b)`。
- **`NumoFFT` シム**: `axis:` は swapaxes して末尾に持ってきて `Numo::Pocketfft.fft` を呼び戻す、`n:` はゼロ埋め・切り詰め、`norm:` は Ruby で掛ける、`fftfreq` / `rfftfreq` / `fftshift` / `ifftshift` は `seq` と `roll` 相当で書く、`hfft` / `ihfft` は Hermitian 拡張してから `irfft` / `rfft` で書く。入力は常に DComplex / DFloat になる(pocketfft の制約)。**シムは 150 行以内**に収め、題材の一部ではなく基盤として test/ で単体に試す(numpy を参照に)。
- スカラーの読み戻しは `FFTBench.scalar` に 1 か所。反復ループ(B・C の時間発展、F のシンボル列)の中では読み戻さない。
- Bit / マスク代入 / NArray 添字は避ける。`max_index` は平坦化した通し番号を返す(E の argmax で踏む)。

### 1.2 Python 側

```python
if os.environ.get("GPU") in ("1", "on", "true"):
    import cupy as xp
    import cupy.fft as xfft
    import cupyx.scipy.signal as xsig       # fftconvolve
    sync = lambda: xp.cuda.Stream.null.synchronize()
    plan_cache = xp.fft.config.get_plan_cache()   # 統計を method.md に
else:
    import numpy as xp
    import numpy.fft as xfft
    import scipy.signal as xsig
    sync = lambda: None
```

- **CuPy のプランキャッシュは既定(デバイスごと 16)のまま**にし、cumo 側の 16 と揃える。`plan_cache.show_info()` の命中数を計測後に出力し、条件ごとに「定常では全部命中している」ことを確認する。
- `CUPY_TF32=0`。FFT には効かないが、D・F の `dot` が絡む経路のために Linalg のときと同じく明示する。
- CuPy と numpy で挙動が違うところ(CuPy は軸の範囲外を黙って通す、など)は `cupy-notes.md` に記録し、題材コードでは踏まない。

---

## 2. 共通の約束事

### 関門(正しさ)

- **DFloat で回す。** 許容誤差は絶対値でなく相対誤差で書く。`‖x − x_ref‖_max / ‖x_ref‖_max ≤ tol`。
- tol は **DFloat 1e-12、SFloat 1e-5** から始め(FFT は O(log N) の誤差成長なので Linalg より厳しくできる)、Numo(pocketfft)で測った実際の誤差の 10 倍に **測ってから** 固定する。表にする。
- **完全一致が取れる題材(A・D・E・F)は完全一致を要求する。** 誤差の議論を持ち込まない。
- どの題材でも **Parseval の等式** `Σ|x|² = (1/N) Σ|X|²` と **往復の恒等** `ifft(fft(x)) = x` を補助の関門として自動で付ける(`gates.rb` の共通関数)。`norm:` の 3 種で Parseval の係数が変わることも関門に含める。
- **Numo と Cumo の直接比較**は `test/parity_test.rb`(numpy 流 API どうし)と `test/pocketfft_parity_test.rb`(`Cumo::Pocketfft` vs `Numo::Pocketfft`: 返る dtype が DComplex / DFloat か、1 次元用関数に 2 次元を渡したときの `ArgumentError`、`fftconvolve` の形がすべて 1 の場合、奇数長)で行う。後者は **cumo の互換層が numo-pocketfft の挙動をどこまで写せているかの直接の証拠**になる。
- 4 実装は **同じ fixture から同じ関門値を JSON に書き出し**、`script/compare_gates.rb` が並べる。

### 計測(速さ)

- narray-llm の `bench/run.sh` を継ぐ: **Cumo / CuPy / 対照(Cumo をもう一度)の 3 系列を 1 ラウンド内でインターリーブ**、順序をラウンドごとに回転、11 ラウンドの先頭を捨てて 10。比は同じラウンド内の比の中央値、後ろに n/10。**対照が 1 をまたがない条件は結果に載せない。**
- **初回と定常を分ける。** 初回 = プロセス内で最初の 1 回(プラン生成 + 作業領域の確保を含む)、定常 = ウォームアップ後の best-of-3。初回は別の表にする。cuFFT のプラン生成は数 ms〜数十 ms で、小さい変換の定常 1 回(数 µs)の 1000 倍になるので、混ぜると何も読めない。
- **キャッシュを溢れさせる条件を 1 つ入れる。** 17 種類以上の形を順に回す条件(G の長さ 3 種 × 窓 × バッチで作れる)で、LRU の捨て方の差が出る。
- CPU 表(Numo / numpy+scipy)は別に取る。GPU と CPU を同じ表に混ぜない。
- best-of-3、ウォームアップ 1 回、区間内に同期を挟まない、区間末で同期(CuPy は `Stream.null.synchronize()`、Cumo は結果のスカラー 1 個を読み戻す)。`Cumo::FFT` は Linalg と違って devInfo を読まないので区間内に同期が無いはず。**nsys で一度数えて method.md に書く**(C2R の退避コピー、swapaxes のコピー、正規化のカーネル本数も一緒に)。
- クロック固定(`nvidia-smi -lgc` / `-lmc`)。fixture の読み込み、プロセス起動は含めない。
- 1 プロセスが 2 秒以上回るように反復回数を条件ごとに決めて `conditions.tsv` に書く。
- **小さい N で GPU が負けるのは正常。** 各題材で **CPU と GPU が逆転する N** を出す。

### fixture

- `script/make_fixtures.rb` が Numo の DFloat で生成し、`data/<題材>_<サイズ>.bin`(`to_binary`、行優先、shape はファイル名)に置く。Python は `np.fromfile(..., dtype='<f8').reshape(...)`。複素数は実部・虚部を別ファイルにして読み側で組む(`to_binary` の複素レイアウトに依存しない)。
- 乱数は Numo の `rand` / `rand_norm` を seed 固定で 1 回だけ。**Cumo でも numpy でも生成しない。**
- サイズは 3 段: `small`(関門、N ≦ 1024)、`mid`(逆転点、N = 2¹¹〜2¹⁶)、`large`(GPU の本領、2D は 4096²、3D は 256³ まで。VRAM 8 GB に収める)。
- **長さは 3 系統を必ず含める**: 2 の冪(radix-2)、2·3·5·7 の積(mixed radix、Whisper の 400 = 2⁴·5² もこれ)、素数(Bluestein)。cuFFT と pocketfft の中の経路が切り替わるので、ラッパーの差とライブラリの差を切り分けられる。

---

## 3. 題材とカバー表

| 関数 | A | B | C | D | E | F | G |
|---|---|---|---|---|---|---|---|
| `fft` / `ifft` | | | | ● | | ● | ● |
| `rfft` / `irfft` | ● | | | ● | | | ● |
| `fft2` / `ifft2` | | | | | ● | | |
| `rfft2` / `irfft2` | | ● | | | ● | | |
| `fftn` / `ifftn` | | | ● | | | | |
| `rfftn` / `irfftn` | | | ● | | | | |
| `hfft` / `ihfft` | | | | | | | ● |
| `fftfreq` / `rfftfreq` | ● | ● | ● | | | | ● |
| `fftshift` / `ifftshift` | | ● | | | ● | ● | |
| `fftconvolve` | | | | ● | | ● | ● |
| `n:` / `s:`(ゼロ埋め・切り詰め) | ● | | | ● | | | ● |
| `axis:` / `axes:`(末尾以外) | | ● | ● | | | ● | |
| `norm:`(backward / ortho / forward) | | | | | | ● | ● |
| バッチ(先頭軸をまとめる) | ● | | | | | ● | ● |

18 関数と `fftconvolve` がどこかで踏まれ、引数の 3 系統(`n:` / `axis:` / `norm:`)とバッチも踏まれる。**A は唯一の実利用者**(narray-llm の Whisper)、**C は 3D と最大サイズ**、**F・G は小さい N の大量呼び出し**で、narray-llm の decode で見えた「固定費の小ささ」が FFT でも出るかを見る場所。

---

## 4. 題材の仕様

各題材は `Problem` クラス 1 つ。`fixture(size)` / `run(dtype)` / `gate(result)` / `bench_step` の 4 メソッドを持ち、`gate.rb` と `bench.rb` はそれしか呼ばない。Python も同じ 4 関数。

### A. Whisper tiny のメル特徴量(STFT → log-mel)

- **入力**: 30 秒・16 kHz の音声(narray-llm の Whisper tiny の fixture と同じもの。無ければ合成のチャープ + 既知のトーン)、メルフィルタバンク行列(80 × 201、Whisper の `mel_filters` をそのまま fixture に)。
- **手順**: フレーム化(n_fft = 400、hop = 160、Hann 窓、反射パディング)→ `rfft(frames, n: 400, axis: -1)` をバッチで 1 回 → パワー `|X|²` → メル行列との `dot` → `log10` → クリップと正規化(Whisper 流)。`rfftfreq(400, 1/16000)` でメル行列の周波数軸を確認。
- **関門**: (1) **narray-llm の Whisper tiny に差し込んで、トークン列が完全一致**(narray-llm 側の既存 fixture `python/fixtures/` の列と)。これはこのリポジトリではなく narray-llm の `test_whisper_*` で確かめ、結果だけ `results/whisper_mel.md` に書く。(2) このリポジトリ内では、`rfft` 経由の log-mel と **DFT 行列の行列積で作った log-mel が DFloat で 1e-10 以内**。(3) Parseval。
- **サイズ**: 3000 フレーム × 400 の 1 条件が本番。計測はフレーム数 300 / 3000 / 30000(10 秒分を繰り返す)。長さ 400 = 2⁴·5² は mixed radix。
- **踏む関数**: `rfft`(バッチ、`n:`)、`rfftfreq`、`dot`。
- **CuPy**: `cupy.fft.rfft(frames, n=400, axis=-1)`。同じ。Whisper 公式(PyTorch)の `torch.stft` と同じ窓・パディングにする。
- **見どころ**: narray-llm の Whisper encode で PyTorch に 15% 負けている件(docs/cumo-history.md)の一部がメル計算なら、ここで取り返せる可能性がある。差し込み前後の encode の tokens/sec を narray-llm 側で測る。

### B. 2D スペクトル法: Taylor–Green 渦(Navier–Stokes 渦度形式、周期境界)

- **入力**: 格子 N × N、粘性 ν、初期渦度 `ω₀ = 2 sin x sin y`(Taylor–Green)と、別に **乱数初期場**(エネルギースペクトルを k⁻⁵ 型に作り込む)。
- **手順**: 渦度 ω を `rfft2` で ω̂ に → 流れ関数 `ψ̂ = ω̂ / k²`(k = 0 は 0)→ 速度 `û = i k_y ψ̂`、`v̂ = −i k_x ψ̂` → `irfft2` で実空間へ → 非線形項 `(u·∇)ω` を実空間で作り `rfft2` → 2/3 則で dealias → 時間積分(RK4、粘性項は積分因子)。`fftfreq(N, 1/N)` で波数、`fftshift` でスペクトルの表示順。
- **関門**: (1) Taylor–Green は非線形項が厳密に 0 になるので、**`ω(t) = ω₀ e^{−2νt}` の解析解に相対 1e-8 以内**(DFloat、t = 1、N = 64)。(2) 乱数初期場・ν = 0 で、**エネルギー `Σ|û|²+|v̂|²` とエンストロフィー `Σ|ω̂|²` が短時間(100 ステップ)で相対 1e-6 以内に保存**(dealias 込み)。(3) Parseval が各ステップで成立。(4) `irfft2(rfft2(ω)) = ω`。
- **サイズ**: N = 64(関門)、256 / 512 / 1024 / 2048 / 4096(計測、1 ステップに `rfft2` 3 回 + `irfft2` 3 回)。
- **踏む関数**: `rfft2`, `irfft2`, `fftfreq`, `fftshift`, `axes:`(`rfft2` の既定 `(-2, -1)` と、転置した配列での明示)。
- **CuPy**: `cupy.fft.rfft2` / `irfft2`。同じ。
- **見どころ**: 1 ステップに 6 回の 2D 変換が同じ形で続くので、プランキャッシュが完全に効く定常状態。ここで CuPy に負けるならラッパーの前後処理(C2R の退避コピー、正規化)が原因。

### C. 3D Poisson(ガウス電荷のポテンシャル)

- **入力**: 格子 N³、辺 L の周期箱、中心にガウス電荷 `ρ = (2πσ²)^{−3/2} exp(−r²/2σ²)`(総電荷 1)。σ ≪ L。
- **手順**: `ρ̂ = rfftn(ρ)` → `φ̂ = ρ̂ / k²`(k = 0 は 0: 周期系では平均を落とす)→ `φ = irfftn(φ̂, s: [N, N, N])`。`fftfreq` 3 本で k。複素版 `fftn` / `ifftn` でも同じ φ を出す。
- **関門**: (1) 解析解 `φ(r) = erf(r / (√2 σ)) / (4π r)`(単位系は ∇²φ = −ρ)との比較。周期像と k = 0 の除去で定数と像の寄与が乗るので、**r < L/4 の 2 点間の差 `φ(r₁) − φ(r₂)` を比べる**(相対 1e-6 以内、N = 64、σ = L/16)。(2) `rfftn` 経路と `fftn` 経路の φ が一致。(3) `∇²φ`(スペクトルで `−k² φ̂` → `irfftn`)が `−ρ` に戻る。(4) `irfftn(rfftn(ρ), s:)` = ρ(**`s:` を省くと奇数長で崩れる**ので、偶数・奇数の両方を踏む)。
- **サイズ**: N = 64(関門)、128 / 256(計測、DFloat は 256³ まで、SFloat は 512³ まで。c128 の 512³ は 2 GB/配列で 8 GB に収まらない)。
- **踏む関数**: `rfftn`, `irfftn`(`s:`), `fftn`, `ifftn`, `fftfreq`, `axes:`(3 軸のうち 2 軸だけ変換する `axes: [0, 1]` も 1 条件入れる → 末尾でない軸の経路)。
- **CuPy**: `cupy.fft.rfftn` / `irfftn`。同じ。
- **見どころ**: 多次元プランと最大サイズ。`axes: [0, 1]` は cumo も CuPy も swapaxes + コピーに落ちるので、そのコピーの速さの比較になる。

### D. 多倍長整数の乗算(FFT 畳み込み)

- **入力**: 2 つの非負整数を基数 B = 10⁴ の桁列(DFloat)として fixture に。桁数 n = 10³〜10⁷。Ruby の `Integer` と Python の `int` で同じ数を復元できる。
- **手順**: (1) `fftconvolve(a, b)`(full)で係数列 → 四捨五入 → 繰り上がり(Ruby 側で `Integer`)。(2) 同じことを `rfft(a, n: m)`・`rfft(b, n: m)`・積・`irfft(…, n: m)`(m = 2 の冪に切り上げ)で手書き。(3) 複素版 `fft` / `ifft` でも。
- **関門**: (1) **Ruby の `a_int * b_int` と桁まで完全一致**(Python は `int` 同士)。(2) 3 経路の係数列が丸める前に 0.5 未満の差。(3) **倍精度で完全一致が保てる桁数の上限を実測で出す**(係数の最大値 `n·(B−1)²` と丸め誤差の関係。n = 10⁷ でも B = 10⁴ なら 2⁵³ の内側だが、FFT の誤差で先に崩れるはず)。SFloat は小さい n で崩れる。**どこで崩れるかを表にして、それ自体を結果にする。**
- **サイズ**: n = 10³(関門)、10⁴ / 10⁵ / 10⁶ / 10⁷(計測)。m は 2 の冪と、あえて 2 の冪でない `n + n − 1` そのまま(Bluestein または mixed radix)の両方。
- **踏む関数**: `fftconvolve`, `rfft` / `irfft`(`n:`), `fft` / `ifft`。
- **CuPy**: `cupyx.scipy.signal.fftconvolve(a, b, mode='full')`、`cupy.fft.rfft(a, n=m)`。
- **見どころ**: `fftconvolve` の実装差(cumo は両方実数なら `rfftn`、CuPy も実数なら rfft 経路)。Ruby の bignum と比べるという関門が Ruby らしい。

### E. 位相相関による画像レジストレーション

- **入力**: 画像 I(N × N、合成のテクスチャ: 乱数を低域通過したもの)、それを **既知の整数シフト (dx, dy) で巡回シフト**した J、別に **既知の小数シフト**で Fourier 位相を回して作った J′。
- **手順**: `F = fft2(I)`、`G = fft2(J)`、`R = F·conj(G) / |F·conj(G)|`、`r = ifft2(R).real`、`argmax(r)` → (dx, dy)。負のシフトを扱うために `fftshift` して中心基準で読む。サブピクセルはピーク周りの放物線フィット。実数入力なので `rfft2` / `irfft2` 経路でも同じ r を出す。
- **関門**: (1) **整数シフトが完全一致**(N = 256、|dx|, |dy| < N/4 の 20 組)。(2) 小数シフトが 0.05 px 以内(放物線フィットの限界。tol は numpy で測ってから固定)。(3) `fft2` 経路と `rfft2` 経路の r が一致。(4) `fftshift(ifftshift(x)) = x`、奇数 N でも。(5) ピーク値が 1 に近い(ノイズ 0 なら 1 − 1e-10 以上)。
- **サイズ**: N = 256(関門)、512 / 1024 / 2048 / 4096(計測、1 回のレジストレーション = `fft2` 2 回 + `ifft2` 1 回 + `fftshift`)。奇数 N(255、1023)も関門で踏む。
- **踏む関数**: `fft2` / `ifft2`, `rfft2` / `irfft2`, `fftshift` / `ifftshift`, `max_index`(平坦化の通し番号 → 2 次元に戻す)。
- **CuPy**: `cupy.fft.fft2`、`cupy.argmax`。同じ。
- **見どころ**: `fftshift` は cumo では `roll` 相当のコピー 1 回で書けるか、CuPy は `cupy.roll`。小さい N では `fftshift` のコストが変換と同じ桁になる。

### F. OFDM 変調・復調

- **入力**: ビット列(fixture)、サブキャリア数 N(64 / 256 / 1024)、QPSK、サイクリックプレフィックス長 N/8、既知の多重波チャネル h(FIR、長さ < CP)、AWGN の雑音(fixture、Eb/N0 の 6 点)。
- **手順**: ビット → QPSK シンボル(N 本)→ `ifft(X, norm: 'ortho')` で時間波形 → CP 付加 → `fftconvolve(x, h)`(mode は full、先頭から切る)+ 雑音 → CP 除去 → `fft(y, norm: 'ortho')` → 1 タップ等化 `Y / H`(`H = fft(h, n: N)`)→ 判定 → ビット。シンボル列(1000 本)をバッチで `[1000, N]` のまま `axis: -1` で 1 回に。`fftshift` で中心周波数順に並べたコンスタレーション図用の配列も作る(関門 (4))。
- **関門**: (1) **雑音なしで BER = 0(全ビット完全一致)**、3 つの `norm:` すべてで(`norm:` を送受で揃えれば結果は同じ、揃えなければスケールがずれて判定は同じだが Parseval の係数が変わる → それを関門 (3) で見る)。(2) AWGN 有りで、BER が QPSK の理論値 `½ erfc(√(Eb/N0))` に対し、1000 シンボルの二項分布の 3σ 以内。(3) Parseval の係数: backward は 1/N、ortho は 1、forward は N。(4) `fftshift` 後の DC が中央に来る。
- **サイズ**: N = 64(関門)、計測は N = 64 / 256 / 1024 × シンボル数 1000 / 10000 のバッチ。**N = 64 × 10000 本が「小さい変換を大量に」の代表条件。** バッチにせず 1 本ずつ 10000 回呼ぶ条件も 1 つ入れる(固定費そのもの)。
- **踏む関数**: `fft` / `ifft`(バッチ、`axis:`、`norm:` 3 種), `fft(h, n: N)`, `fftconvolve`, `fftshift`。
- **CuPy**: `cupy.fft.fft(x, axis=-1, norm='ortho')`、`cupyx.scipy.signal.fftconvolve`。同じ。
- **見どころ**: 1 本ずつ 10000 回の条件は、narray-llm の decode と同じ「起動律速」の構図。cumo の固定費(2 µs 台)対 CuPy(3.7 µs 台)が FFT でも出るか。プランキャッシュの命中は両方 100% のはず(確認する)。

### G. 相関・スペクトル推定(Wiener–Khinchin、Welch 法)

- **入力**: 既知の正弦波(f₀、振幅 A)+ 白色雑音(σ²)の実信号 x、長さ N は **3 系統**(2¹⁶、2⁴·3³·5²·7 = 75600、素数 65537)。別に短い x(N = 256)で直接計算との比較用。
- **手順**: (1) 自己相関 `r = irfft(|rfft(x, n: 2N)|², n: 2N)[0...N]`(Wiener–Khinchin)。(2) 同じ r を `fftconvolve(x, x.reverse)` から。(3) 短い x では O(N²) の直接計算。(4) r は実・偶対称なので、**`hfft(r[0..N/2], n: N)` で実数のスペクトル**を出し、`rfft(x)` のパワーと一致させる(hfft の自然な用途)。`ihfft` でその逆。(5) Welch 法: 長さ L のセグメント、Hann 窓、50% 重なり、`rfft(segments, axis: -1)` バッチ、平均 → PSD。`rfftfreq(L, 1/fs)` でピーク周波数。
- **関門**: (1) 短い x で FFT 経路の r と直接計算が相対 1e-10 以内。(2) `fftconvolve` 経路と一致。(3) `hfft` のスペクトルが `|rfft(x)|²` と一致、`ihfft(hfft(h)) = h`。(4) Welch の **ピーク周波数が f₀ の bin に一致**、ピーク高さが `A²/2 · (窓の係数)`、雑音床が `σ²/fs` に 10% 以内(統計)。(5) 3 系統の長さすべてで (1)〜(4) が通る(**Bluestein 経路の正しさ**)。
- **サイズ**: N = 256(関門)、計測は 3 系統 × セグメント長 L = 256 / 1024 / 4096。**17 種類以上の (L, バッチ) の組を順に回す条件**を 1 つ作り、プランキャッシュの LRU を溢れさせる。
- **踏む関数**: `rfft` / `irfft`(`n:`), `hfft` / `ihfft`, `rfftfreq`, `fftconvolve`, `fft` / `ifft`(複素版の確認)。
- **CuPy**: `cupy.fft.hfft` / `ihfft`、`cupyx.scipy.signal.fftconvolve`、Welch は自前(`cupyx.scipy.signal.welch` は使わず、Ruby と同じ手順で書く)。
- **見どころ**: 素数長で cuFFT と pocketfft の中の経路が両方 Bluestein になる。ここで cumo 対 CuPy の比が他の長さと同じなら、差はラッパーで説明できる。キャッシュ溢れ条件で、cumo の LRU(Event 待ちで破棄)と CuPy の LRU の破棄コストの差が出る。

---

## 5. CuPy との違い(docs/cupy-notes.md に育てる)

CuPy 14 系の `cupy/fft/` を読んで確認したもの。**関数の欠けは無い**ので、Linalg の `cupy-gaps.md` とは性格が違い、挙動差と実装差を記録するファイルになる。

| 項目 | CuPy | Cumo::FFT(計画) | bench での扱い |
|---|---|---|---|
| 関数 | `cupy.fft` 18 関数、`cupyx.scipy.fft` に `dct` / `dst` / `fht` も | numpy.fft の 18 関数。`dct` / `dst` は無い | 題材は 18 関数の範囲で書く |
| `fftconvolve` | `cupyx.scipy.signal.fftconvolve(mode, axes)` | 互換層に `fftconvolve(a, b)`(full のみ) | mode は full だけ使う |
| プランキャッシュ | デバイスごと LRU 16(`cupy.fft.config`) | スレッドごと・GPU ごと LRU 16 | 既定どうしで比べる。溢れ条件を G に |
| 作業領域 | CuPy のメモリプールから | cumo のメモリプールから(ストリーム別アリーナ) | 同じ土俵 |
| 正規化 | 変換後に乗算 1 パス | 同じ | 同じ |
| 末尾でない軸 | swapaxes + コピー | 同じ(advanced layout は後の最適化) | C の `axes: [0, 1]` で比べる |
| 軸の範囲外 | 黙って通す | `DimensionError` | 題材では踏まない。notes に記録 |
| 長さ 1 の `irfft` | `CUFFT_INVALID_SIZE` | `ArgumentError` | 同上 |
| 単精度 | float32 → complex64 | SFloat → SComplex | 同じ |
| 半精度 | 2 の冪のみ対応 | HFloat は SComplex に昇格して変換 | 計測しない(方針が違う) |

**Numo 側(numo-pocketfft)との違い**は `pocketfft_parity_test.rb` で機械的に確かめ、`results/pocketfft-compat.md` に表で残す(dtype、例外の型、`fftconvolve` の端、奇数長)。

---

## 6. 段階

### 第 0 段階: 計画 — このファイル

### 第一段階: 足場(Numo だけで通す)

- `backend.rb`、**`numo_fft.rb` シム**、`fixtures.rb`、`gates.rb`(相対誤差、Parseval、往復恒等、完全一致)、`timer.rb`(初回/定常の分離)、`Problem` の基底クラス、`gate.rb` / `bench.rb` の骨組み
- シムの単体テスト: numpy で作った参照(fixture として `data/shim_ref_*.bin`)と、`n:` / `axis:` / `norm:` / `hfft` / `fftshift` の全組み合わせで一致
- `make_fixtures.rb` が 7 題材の `small` fixture を作る(複素数は実部・虚部を別ファイル)
- Python 側の `backend.py` / `fixtures.py` / `gates.py` と、fixture を読んで **バイト一致**することのテスト
- `bench/run.sh` を narray-llm から移し、`conditions.tsv` に「初回/定常」の列を足す

**受け入れテスト**: シムが numpy 参照と一致。7 題材の `small` fixture が Ruby と Python で同じ値に読める。`gate.rb` が「題材なし」で空の JSON を出す。**GPU 無しで全部通る。**

### 第二〜第八段階: 題材 D → E → G → F → B → C → A の順に 1 題材ずつ

順序の理由: **D と E は 1 次元・2 次元の基本関数だけで完全一致の関門が取れる**ので、`Cumo::FFT` の第 1 段階(`fft` / `ifft` / `rfft` / `irfft`)と第 2 段階(2D)が出た直後に使える。**G は `hfft` と長さ 3 系統**で cumo 側の第 3 段階を待つ。F は `norm:` とバッチ。B・C は反復と 3D で重く、A は narray-llm 側に手を入れるので `Cumo::FFT` が固まってから最後。

各題材で:
1. Ruby 実装(`XM` / `XFT` / `XF` だけで書く)と関門。**Numo で関門が通る**ことが最初の受け入れ
2. Python 実装(numpy / scipy)。**Ruby と同じ関門値が JSON で一致**
3. CuPy で動かす(`GPU=1 python`)。挙動差があれば `cupy-notes.md` に書く
4. Cumo で動かす(`GPU=1 ruby`)。**cumo 側の該当段階が未実装なら `NotImplementedError` で skip** し、そのことを結果ファイルに書く(bench は cumo を待たない)
5. `parity_test.rb` に Numo シム vs Cumo::FFT の直接比較を足す(該当関数だけ)
6. `mid` / `large` fixture、`conditions.tsv` に初回/定常の条件を足す、`bench.rb` で 1 条件だけ測って所要時間が 2 秒以上になる反復回数を決める
7. `docs/results/<題材>.md` に関門の値と最初の数字

**受け入れテスト(各題材共通)**: Numo で関門が通る。numpy で同じ関門値。CuPy で関門が通る。Cumo で関門が通る(または未実装 skip)。parity が通る。

### 第九段階: `Cumo::Pocketfft` 互換層の試験

- `pocketfft_parity_test.rb`: 13 関数 × (dtype 4 種の入力、1 次元 / 2 次元 / 3 次元、空、奇数長)で `Numo::Pocketfft` と `Cumo::Pocketfft` の **返る dtype・例外の型・値** を突き合わせる
- `fftconvolve` の端: 形がすべて 1、片方が複素、奇数長 × 偶数長
- 結果を `results/pocketfft-compat.md` に表で

### 第十段階: 計測の本番

- 7 題材 × サイズ × dtype(SFloat / DFloat)× 初回/定常を `run.sh` で回す(Cumo / CuPy / 対照)
- CPU 表(Numo / numpy+scipy、DFloat のみ)を別に
- 逆転点の表、プランキャッシュの命中率(CuPy は `show_info()`、cumo は統計 API があればそれ、無ければ nsys のプラン生成カーネルの有無)
- **対照が 1 をまたがない条件を除いて** README に載せる
- nsys で各題材の代表条件について **区間内のカーネル本数・コピー本数・同期回数**を Cumo と CuPy で数え、method.md に書く
- A の差し込み結果(narray-llm の Whisper encode の前後)を narray-llm 側の results に書き、ここからリンク

### 第十一段階: 見つけたものを cumo に戻す

- `cumo-issues.md` に、性能で負けた条件と原因の仮説、棄却した仮説
- 候補としてあらかじめ見えているもの: 末尾でない軸の advanced layout(C)、`fftshift` のコピー削減(E)、正規化のコピーへの畳み込み(B)、LRU の破棄コスト(G)

---

## 7. 数字の出し方(README の表の形)

題材ごとに 1 表(定常)。行はサイズ、列は Cumo / CuPy / Cumo÷CuPy(n/10)、SFloat と DFloat で表を分ける。初回(プラン生成込み)は題材横断で 1 表。

```
### F. OFDM (fft + ifft + 等化、シンボル 1 本あたり µs、定常、小さいほど速い)

| N × バッチ | Cumo | CuPy | Cumo / CuPy |
|---|---|---|---|
| 64 × 1(1 本ずつ) | ... | ... | 1.xx 10/10 |
| 64 × 10000 | ... | ... | ... |
| 1024 × 1000 | ... | ... | ... |
```

比は「CuPy ÷ Cumo」で **1 を超えれば Cumo が速い**(narray-llm と同じ向き)。CPU 表は Numo / numpy+scipy で同じ形。

---

## 8. リスク・先に確認すること

| リスク | 対応 |
|---|---|
| cumo 側の `Cumo::FFT` が題材より遅れる | bench 側は Numo / numpy / CuPy の 3 者で先に完成させる。Cumo は `NotImplementedError` で skip。**bench は cumo を待たない** |
| Numo シムが numpy と合わない(pocketfft の `irfft` の長さの既定、など) | 第一段階で numpy 参照の fixture と全組み合わせで突き合わせる。合わない点は `numo_fft.rb` のコメントと `method.md` に書く |
| numo-pocketfft が倍精度しか無い | CPU 表は DFloat だけ。SFloat の CPU 列は numpy のみ。README にそう書く |
| 初回と定常の混在 | `timer.rb` で分離し、`conditions.tsv` に列を持つ。初回は別表 |
| A の narray-llm 側の変更 | narray-llm の `Ops` に `XFT` を使う経路を足す PR を別に切る。bench 側からは結果へのリンクだけ |
| C の VRAM | DFloat は 256³ まで、SFloat は 512³ まで。`large` の条件を VRAM 8 GB で決めたことを method.md に |
| D の完全一致が崩れる桁数 | 崩れる点を探すのが目的なので、崩れたら表に書く。tol の緩和はしない |
| 素数長(Bluestein)が cuFFT で極端に遅い | それ自体が結果。G の表に 3 系統を並べる。CuPy も同じ cuFFT なので比は同じ土俵 |
| `fftshift` の実装が cumo に無い段階 | シムと同じ `roll` 相当を題材側の helper に持ち、cumo 側に入ったら差し替える |
| CuPy のプランキャッシュが別スレッド・別デバイスで分かれる仕様 | 計測は 1 スレッド・1 GPU。method.md に書く |

---

## 9. 実装セッションへの依頼事項

1. 第一段階を Numo だけで通す。**GPU 無しでも `rake test` が全部通る状態を最後まで保つ。**
2. 題材は D → E → G → F → B → C → A の順。**前の題材の受け入れテストが通るまで次を書かない。**
3. 題材コードに `Numo::` / `Cumo::` / `Pocketfft` / `numpy.` / `cupy.` を直接書かない(`XM` / `XFT` / `XF`、`xp` / `xfft` / `xsig` だけ)。
4. 関門の tol は「測ってから固定」。完全一致が取れる題材(A・D・E・F)では tol を持ち込まない。
5. 計測は初回と定常を必ず分ける。定常の表にプラン生成を混ぜない。
6. 計測の主張は run.sh の 10 ラウンド + 対照系列付きでのみ。単発の数字を README に書かない。
7. cumo で見つけた問題は `cumo-issues.md` に、棄却した仮説も含めて残す。CuPy との挙動差は `cupy-notes.md` に。
