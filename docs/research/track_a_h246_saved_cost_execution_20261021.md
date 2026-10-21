# H2/H4/H6高次PF：保存値統計・総費用の限定実行契約

2026-10-21。利用者の保存値科学数値解析one-shot承認と、実行前STOP・§3採択案に対する「続きを行って」を、本outer32系列だけの条件付きshot政策による解析再開指示として結合する。過去のpolicy未採択記録・STOP・無効grantは変更しない。共通v2全体やm5主比較採用AD03の変更ではない。

## 入力と政策

固定結果H2 `a05f4ed19e87e1eb65f7fc346385b5b4eae69845`、H4/H6 `605ca07a653241db49348fbd7f4e091f9f18ced7`。科学SOURCEはそれぞれ`4ec26921…`/`5a448d94…`。保存条件は直鎖1 Å、STO-3G、tol-only1e-8 DF3/10/19、prefix1/3/5、T=.8/P=0、S2/Yoshida4/m5_best、q1/2/4/8、K4/outer32、symmetric-directional・scalar once、保存compilerである。

[前報§3](track_a_h246_higher_pf_saved_evaluation_preflight_STOP_20261021.md)の採択案をこの数値診断へ適用する。5精度.05/.01/.005/.001/.0005、ε_axis=ε/√2、α_axis=.025、経験的u倍率1、保存log_B+保存log_B_margin、u/headroom≤.01。`d0_frozen_shot_rule_v1.py:u_aware_shots`のbinary64演算順・整数ceil・LOG_DOMAIN_ONLY/nullを維持する。参照精度条件付きであり、厳密bias/normalization上界・母平均保証ではない。

通常signedと非相殺を別区分とする。PRの非相殺は|outer_PF|+|finite_RTE|。full PFは|PF|。B0/B0Iの保存値はdiscard+PFが合計されているため、個別成分がない非相殺値を復元せずnullとする。通常signedの保存適格性を再現gateとする。

## 一回予算・計算・照合

新探索標本だけを使う。H2 525/H4 793/H6 607本、合計1925、PR36群。H4/H6のm5決定論64候補を保存signalへ結合し、うち48候補96軸の実測単価を使える範囲だけ会計する。計100候補×5精度×2bias区分=1000候補精度記録、凍結shot pair評価は最大1000回、系/精度/biasのまとめ30件、高次PF対S2の比較は最大60件。未取得・不適格・成分不足もnull/status行で保持する。

整数RZの平均・不偏分散・cos/sin共分散を有理数で計算する。G_i=N_cos C_cos,i+N_sin C_sin,iの平均・分散・SEを計算し、共分散を含む分散恒等式を有理数で完全一致させる。標準library統計関数と中心化二段計算で別照合する。shotはDecimalの基本演算を各段階binary64へ丸める別経路で整数完全一致を要求し、sqrt/exp/logのlibm値を共有する点は明示する。これは独立した数値経路であり、別AI/独立研究者のレビューではない。

整数shot・有理数恒等式はexact一致、浮動統計の照合は256×binary64 epsilon×max(1,絶対値)を技術的roundoff判定とする。科学的bias allowanceやcompiler閾値を変更しない。

式の最良は登録4q内の探索点最良、同値候補は保持する。未知適格性・欠測費用を含む集合は最良をnullとし、0補完しない。高次PF対S2の総費用比をshot総数比×shot重み付き有効単価比へ正確に分解する。独立候補cohortの差SEでは候補間独立を仮定するが、候補間paired構造・選択後補正・母平均順位保証を主張しない。

## 決定論的対照と限界

m5対照は保存conditionsのHamiltonian/state/snapshot binding、compiler、preparationをsignal/PRへ結合した同native系列の部分集合として比較する。新92/保存再利用4軸を来歴で区別し、model推定と呼ばない。旧強化S2/S4のnative/preparation/runtime完全identity結合は本系列で未確立なので、旧対照を置換せず、m5部分集合の最良を全強化PF最良と表示しない。H2には新系列に同じidentityで結合した決定論signal集合がなく、対照比較はnullとする。H4 legacy DF12、別788軸研究、旧n8、pilot、確認cohortを混合しない。

未観測S0/S2_single/S2_multi/S4の条件付き費用・母分散上界はunknown。単価分散0や標本SE0は母分散0・rare費用0の証明ではない。独立確認対象の候補と判断材料は探索点比較として提示し、新n_Cの設計・sampling/compileを自動実行しない。

## 認可・実装・STOP

専用preparerが現在の政策採択記録、SOURCE/input/runtime、plan、科学call予算を結合した新しい有効grantを発行する。runnerは入力/SOURCE/hashを検査した後に未使用rootへCONSUMEDを排他的に作り、候補ごと非返金ledgerを予約する。独立候補を利用可能CPUに応じたthread poolで実行し、元入力は一度だけ読んで全5精度・両bias区分へ共用する。新sampling/signal/simulation/回路生成/compile/fitは0。

SHA・native受入・cohort・保存適格性・独立数値経路の不一致、異常exitでSTOPし、同grantのretry/resumeは禁止。source/入力・旧raw/grant/ledger/marker/STOPは変更しない。共有設定や他jobは変更しない。

人工22件PASS。数値解析と終了監査の後、結果・SOURCE/SHA・grant・ledger・終了証拠を保存し、local commit後mandatory STOP。公開push、新sampling/compile、独立確認P、AD03採択、詳細科学レビュー、論文本文は行わない。

次の担当Codex：封印と一回解析・終了監査・報告。次の担当ユーザー：報告を見て独立確認対象と目的を判断する。
