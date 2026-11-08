# O5 C1：固定q2_mixedの最小head-tail人工検証PLAN

2026-11-08。利用者の一括実行指示の範囲だけを専用branch/worktreeで実行する。承認証明やgrantを新設しない。基点62c7b578。数学条件・SOURCE・入力・精度を以下と[PLAN.json](PLAN.json)で固定し、実行SOURCEをcommit/push・照合してから一回だけ起動する。

## 固定範囲

C7-2 CASESのq2_mixed/cut1、2-mode/3-block、frame角.23/−.37/.41、scalar .17/identity .04、h=.1/τ=.2/K0、(plus,zz)→(minus,plus)。分子geometry/AO basis/DF rank/L_Dは非該当。主接口はhead block2→tail frame1、同frame2は対照。一種類だけをStrong/Reference両方へ適用する。tail→head、tail異frame run全体、mode中心合算、step/global-cut/lane越しは変更しない。

既存StrongBuilder.transition/actual basis/gauge、C3 mixed_head/physical_segment/prepare_steps、full-frame Provider、Reference gadget/control_one、C7-2 exterior lift、既存O5人工raw/postcheckとP0/MPFR160を再利用する。元科学codeのファイルは変更しない。新adapterは意味の分かった接口のnative命令列/wiresを厳密照合し、取り除くbasisのdeclared phaseを差し引き、relative橋のdeclared phaseを一回加える。basis IDや近似rotationだけで削除しない。

C7-2 make_casesは無条件に54回路を構築するため今回呼ばず、既存context/event/exterior liftで指定一caseだけを組み立てる薄い入口を置く。既存C3をプロセス内observerで呼び、事前に一回生成したtail2個をそのまま返して重複emissionを避ける。patchはこの専用process内だけで直ちに復帰し、他jobや共有設定に影響しない。既存checkerは複製せず呼ぶ。

## 検証順序と一回性

1. 局所正方向G_T†G_Hの全Fock unitary/independent exterior/vacuum位相、同frame対照を検証する。N1逆方向、N2 gauge欠落、N3 ordinary→directional、N4同IDのrotation/wire/gauge不整合、N5介在Xを検出する。η=π/7のphase witnessは別のlocal builder cacheだけへ設定し、whole-path inputを変えない。
2. local PASS後、既存C3でS0/R0を構成し、意味上の接口だけをspliceしたS1/R1を作る。A/Bの変更後にactual A.inverse()を取得する。AB=V、両half vacuum、scalar/identity/χ、Strong branches/coherence、全4入力と全reference出力、X/Y、全unitary位相を既存checkerで照合する。
3. 全raw PASS後だけS0/S1/R0/R1×X/Y、最大8 wrapperの共通P0へ進む。既存C7-2の4 baselineはsource/profile/hash/canonical raw IR/full phase/wire/旧postcheck PASSを満たす場合だけ再利用する。同一IRはcache。新compileは最大8で、再利用できれば通常4以下。独立compileは実affinity/空きRAMと非干渉を考慮し並列化する。小さい人工4×4/32×32行列を使い、資源の低い固定上限は設けない。
4. 全8のraw/compiled full-unitary phase、X/Y/coherence/χ/reference leakageを照合し、7資源とS1−S0/R1−R0/前後R−S/差の差を別々に記録する。局所価格を足して利益を作らず、不利な結果も保存する。

component1e−10・observable Frobenius/signal2e−10・rtol0・phase fitなし。P0はQiskit1.3.0/opt1/seed17/rz,sx,x,cx/approx1/initially_zero=False/topology-free/MPFR160不変。正当なFAIL後に閾値・fixture・cut・modeを変更しない。技術的不具合は旧SOURCE/partialを保存し、許可された同じscopeの未完了処理だけを修正して継続する。

## 保全・公開・STOP

namespace artifacts/track_d_head_tail_artificial/2026-11-08/run01。原本QPY/運用情報はlocal専用、公開は既存C7-2と同じ科学scalar/compact構造/SOURCE/hash/STOP。元main/他worktree/C6a/C7/C8-1/O2のSOURCE/raw/result/grant/ledger/marker/STOPを無変更とし、q64進捗/未完了出力へアクセスも操作もしない。共有OS/scheduler/venv/cache/他ユーザーの設定は変えない。

人工結果を整理し、改善時は未実行の最小H4移送案、増分0/悪化時はC1を区切りO1/O2/O3共同最適化へ重点を戻す案を作る。新H4/新sampling/fit/N/全13round/詳細科学レビュー/C2は行わない。専用branchへ公開しremote SHA/tree/public bytesを照合後mandatory STOP。次の判断はユーザー/GPT。
