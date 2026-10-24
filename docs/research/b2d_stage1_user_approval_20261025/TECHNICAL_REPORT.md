# B2-D Stage 1：通常ユーザー承認方式への限定修正

2026-10-25 JST。**技術準備完了、科学実行未承認。**
実承認記録・正式grant発行／claim／消費、Hamiltonian／operator／signal数値評価、Gaussian回路生成、native build／compile、保存結果の再解析は実施していない。
発行の動作確認は、隔離tmp・MOCK表示の人工契約／承認だけで行った。

## 固定版と成果物

- 起点：文書公開 `ceb4c50a23c57743343e66cfc6aba69a9a4df3b8`。旧科学SOURCE `348e08cbc44fb0a81cfc4b6932914f7b146a5b9c` を履歴として保全。
- 新科学SOURCE：[91f45aeb50275bb6d22050a304d0a9b1ff025d70](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/91f45aeb50275bb6d22050a304d0a9b1ff025d70)。専用branch `codex/research-b-b2d-user-approval-20261025`。
- [新契約・ユーザー操作](CONTRACT.md)と[未承認JSON](contract.proposal.json)。採択後digestは `155728667abd5a663b44bc06b4e596ea6d64d529da6b02ba1be15ee8c050d4fb`。旧26aaf07c…と区別する。
- 科学checkout `/tmp/prt-b2d-stage1-user-science-20261025` は新SOURCEのclean detached worktree。未承認draft `/tmp/prt-b2d-stage1-user-draft-20261025` にproposalと未承認sampleのみを生成。

## 最小変更と再利用

| 対象 | 今回の変更 |
|---|---|
| manual_authority.py | root必須・SUDO_UID／GID・chown・/var/lib記録を撤廃。通常ユーザーの0700専用directory／0600記録、同一UID、symlink／公開権限拒否。承認方式／記録先／UIDと同一アカウントの限界を契約へ束縛 |
| 発行処理 | 既存packet、canonical/hash、receipt項目を再利用。発行前にsource closure・recipe／layout・scope・環境・fresh outputを確認。既知sample・空文・不一致・再発行を拒否。科学処理・claimなし |
| prepare／issue／run入口 | 未承認proposalへapproval_policyを追加。CLIは既存引数を維持し、表示を通常ユーザー方式へ更新 |
| preflight／launcher | 固定manual providerの既存照合を再利用。root専用Git指定を削除。launcherの実行処理は変更なし（説明文のみ変更） |
| 人工test | 旧「非root発行拒否」「user所有receipt拒否」の2件を新方式の正常系／拒否系へ置換。focused test6件を追加 |

InputRecipe／loader、5方式のDAG、DenseBackend、PF係数、finite RTE／normalization、scope登録、A4数学参照／published列gate、失敗保存、atomic claimは既存SOURCEを再利用した。
科学・scope・環境・既存claim関連10ファイルについて旧科学SOURCEとのbyte一致を確認した。既存trotterlibの科学関数も変更していない。
新旧proposalのcall_scope・recipe・layout・environment_binding・依存version・Python runtime・atolは完全一致した。36行・941 dispatch・448 exponential key、内部Gaussianを含む入力scope、published gate4件を保全。
汎用認証システム、外部API、鍵管理、grant DBは追加していない。

## 人工テストと環境

変更前74件PASSを確認後、同じ5ファイルを独立processで並列実行した。固定venvの`-I -B`と科学import barrierを使用。

| テスト | 件数 | 結果 |
|---|---:|---|
| tests/management/test_b2d_adapter.py | 15 | PASS |
| tests/management/test_b2d_evaluator.py | 14 | PASS |
| tests/management/test_b2d_input_preflight.py | 17 | PASS |
| tests/management/test_b2d_stage1_launcher.py | 16 | PASS |
| tests/management/test_b2d_execution_ready.py | 18 | PASS |
| 合計（既存74件を適合更新＋追加6件） | **80** | **PASS** |

再実行commandは各ファイルに対して `/tmp/pr-phase2df1pre-private-20261023/venv/bin/python -I -B tests/management/<filename>`。
通常UID1001で実際のCLIをrunpyから呼び、mock source確認・mock環境確認の下で隔離人工packetを発行し、receipt binding／0700・0600／未claimを確認した。
新focused確認はsample／空承認、SOURCE・scope・recipe・layout・環境・policy・contract digest・retry／STOP不一致、fresh output、再発行、CLI手動確認必須を含む。
既存testで未承認拒否、任意authority拒否、使用済みgrant／O_EXCLによる二重claim拒否、scope実績、gate先行と失敗STOPも確認。
無印の偽文がユーザー由来かをテストで証明したとは扱わない。人工承認は実承認の取得ではない。
実array／Hamiltonian／Gaussian／science backendを呼ばず、既存science suiteも起動していない。人工PASSは科学correctnessや優位性の証拠ではない。

既存CPython3.11.9／50packageのenvironment bindingをmetadata・find_spec・requirementsだけで再照合し、不変だった。共有venv・依存version・global設定は変更なし。
32 CPU、RAM約62 GiB・available約52 GiB、GPU driver利用不可を確認。人工testは5ファイルの独立並列、任意のCPU／RAM／worker上限なし。実科学の数値高速化変更や速度倍率の主張なし。
共通運用指示2026-10-23正式版を実確認し、SHA-256 `7ce342dbf7086424f1afd837d8c274687bc28e4838601f5ad96f5b626d95667f` に一致。

## 安全性・限界・保全

本人による研究Bチャット承認→本人の原文保存／通常権限での手動発行→Codexへの別実行指示、の運用である。
private所有権は他OSユーザーへの制限であり、同じOSアカウントのCodexを本人から識別する独立認証ではない。Codexの実承認生成・正式発行・sample転用禁止は会話上の運用規則である。
Hashは内容・条件を束縛し、本人性を証明しない。実承認・別実行指示は科学実行前に会話上でも確認する。

旧frozen checkoutは348e08のclean状態、旧draft bytesは不変。旧正式packet／root承認先と新ユーザー承認先は未作成、旧・新科学専用artifact directoryも未作成。
既存SOURCE、raw結果、grant、ledger、消費marker、STOPへの書込なし。他研究・C1へ干渉せず、既存環境も保全。
科学結果・研究判断・evidence statusの変更はないため、研究概要・規範文書・validation manifestの科学的主張は更新しない。技術準備の記録と索引だけを追加する。

## 残件・次の担当

技術的な独立blockerは残していない。ユーザーが[新契約](CONTRACT.md)を研究Bチャットで明示承認し、自身の端末で実承認原文を保存・正式発行した後、Codexへ別途一回の科学実行を指示する。
科学入力identity・N=1非可換性・operator同値性・complex128識別力は未検証のまま、将来の固定gateへ残す。不一致時に条件を探索せずSTOPする。
専用branchだけをcommit／pushし、remote SHAと公開された変更ファイル全件を照合する。**公開確認後mandatory STOP。正式発行・科学実行へ自動進行しない。**
