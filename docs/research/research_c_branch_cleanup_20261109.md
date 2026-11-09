# 研究C Git branch整理・履歴保全（2026-11-09）

**棚卸し・履歴保全・ローカル1本の削除を実施。remote 58本の削除は認証不足で失敗し、保留として残した。remote整理完了とは扱わない。** 科学計算・数値再解析・重要GPT科学レビューは0件。

## 範囲と集計

観測時刻: `2026-11-09T07:47:18.731274+00:00`。対象は`codex/research-c-*`と、Track Aとの共用branch `codex/track-a-research-c-integration-assessment-20261107`。全HEADのsubject照合でも別名の新たな研究C対象は確認されなかった。

| 項目 | 作業前 | 作業後 |
|---|---:|---:|
| 全local branch（実在ref） | 634 | 636 |
| 全remote branch（実GitHub ref） | 405 | 407 |
| 研究C local | 25 | 25 |
| 研究C remote | 80 | 81 |
| 研究C branch名の和集合 | 83 | 84 |
| remote tag | 103 | 103 |

本作業の差分はlocal 1本削除・local 1本追加、remote 1本追加・削除0本。別チャット等による対象外のlocal追加2本、remote追加1本も観測したため、全体集計の増分と本作業の増分を分ける。対象外の追加は操作していない。

対象外local追加: `codex/pr-reference-v6p-prep-20261109`, `codex/track-d-consolidated-20261109`。remote追加: `codex/track-d-consolidated-20261109`。既存local/remote HEADの移動、既存remote tagの削除・移動は観測0件。

初期remote-tracking refは1,345本で、実remote 405本とは異なる。stale refの一括pruneや、tag・GC・reflog期限・他worktreeの変更は行っていない。

## 分類と実行結果

全83本の名前、local/remote別の元full SHA、目的、最終成果物、固有commit数、SOURCE/結果/STOPの有無、保持refと到達可能性、worktree、PR、分類、実行結果は [branches.csv](research_c_branch_cleanup_20261109/branches.csv) に記録した。作業前分類と実行後分類を別列に残した。

| 分類 | 事前判断 | 実行後 |
|---|---:|---|
| KEEP | 最新REPORT 1本 | 既存1本＋新規の保全・整理branch 1本 |
| DELETE_MERGED | 15本 | local 1本削除。remoteは全15本とも認証失敗でHOLD |
| PRESERVE_THEN_DELETE | 43本 | 履歴保存は公開・照合済み。remoteは全43本とも認証失敗でHOLD |
| HOLD_UNRESOLVED | worktree使用24本 | worktree 24本＋remote認証失敗58本＝82本 |

KEEP:

- `codex/research-c-three-method-independent128-integration-20261108` — `776e0a16ac98762631fe9307672725fdfee3b73d`。最新統合REPORT。
- `codex/research-c-branch-cleanup-20261109` — 本報告・履歴保全の保持ref。現在の公開HEADはbranchを参照。固定の履歴保存commitは`054fa91493713fbf8d233828e9070dc07b8c2a1c`。

削除したlocal branchは次の1本。worktreeへのbindingがなく、公開保持refから元HEADを含む全祖先に到達可能であることを削除直前に確認した。`git branch -d`を使用し、元branchのremoteは認証失敗により残っている。

| local削除branch | 元HEAD | remote結果 |
|---|---|---|
| `codex/research-c-h4-relocation-preparation-20261106` | `052941ab7ce680f57b0b5ba9ef27ef8cdd4937c1` | 認証失敗・保持 |

worktree使用中の24本は下表。未commit変更・raw・cache・jobの有無を理由に解放する操作はせず、worktree自体を保持した。Track A共用branchもこの24本に含む。

| HOLD branch | 元HEAD |
|---|---|
| `codex/research-c-baseline-comparability-20261106` | `77ded5c12db21572933653cb26758436da3d9ecd` |
| `codex/research-c-existing-df-improvement-overlap-20261107` | `fd0dd0704ff3172c88c598aab91d63dbe2729c62` |
| `codex/research-c-final-evidence-synthesis-20261107` | `3eccab64a2672813e479c113c2485219f8479885` |
| `codex/research-c-h4-100-paper-integration-20261106` | `5a53d0569b70e4417765c4876616260db911e603` |
| `codex/research-c-h4-relocated-downstream-execution-20261106` | `a67036a6baafc00deaa38a1b0b801d313d519ad5` |
| `codex/research-c-h4-relocated-downstream-preparation-20261106` | `a67036a6baafc00deaa38a1b0b801d313d519ad5` |
| `codex/research-c-h4-relocated-handoff-20261106` | `3015d4fb342d4d98b58e41d57553e75539a07655` |
| `codex/research-c-h4-relocated-process-parallel-20261106` | `6301b630569a9371db26ff511fde819c6fe97060` |
| `codex/research-c-h4-relocation-execution-20261106` | `d601e0ef4375c37185211aa75685ae0e582f3e64` |
| `codex/research-c-h4-strong-d-relocation-execution-20261107` | `91417fb3ef75e7eec0849ad39f0649d170754664` |
| `codex/research-c-h4-strong-d-relocation-preparation-20261107` | `e6652ab841aae5ae700286bf1ab249ad4fb85461` |
| `codex/research-c-h6-conditionfix-20261106` | `d166c75c407f4b2f74f239018fe4bbc5cf9dd390` |
| `codex/research-c-h6-linesearchfix-20261106` | `d7ea73b56bd2cac1e85a658de0c2848e10db658f` |
| `codex/research-c-h6-parallel-technical-20261106` | `a96672cbd4d57756c54bf9da5a6d18d5ed1eac09` |
| `codex/research-c-h6-runner-completion-20261106` | `30c6f47529d708a0dd1f67818fb455325b084fc8` |
| `codex/research-c-h6-stablemetric-20261106` | `0e74b88f5734bb9b5459497c789fb2a5e008910d` |
| `codex/research-c-h6-trustmodel-20261106` | `f97c2257d131c6196601803ed442ed87a7708ae6` |
| `codex/research-c-h6-validation-preparation-20261106` | `68634ef468eb6c912de7bf569484fc454fb32548` |
| `codex/research-c-integrated-cgfix-20261106` | `a7face80e10fd74e1a49dc530409db5b7f677aaa` |
| `codex/research-c-integrated-picklefix-20261106` | `417fa459b823509be5c2010c5836e58955088282` |
| `codex/research-c-integrated-server-20261106` | `a7face80e10fd74e1a49dc530409db5b7f677aaa` |
| `codex/research-c-strong-d-alignment-20261106` | `bfc32c92a9c7a028e2d9065b71b4258809f53c0f` |
| `codex/research-c-three-method-z-relocated-comparison-20261108` | `709813805d41e470676e311cd63a740af23bf33c` |
| `codex/track-a-research-c-integration-assessment-20261107` | `f2e3df45043e874a5a182d856c21c9d96b4073d3` |

## 履歴保全の内容

保持ref `codex/research-c-branch-cleanup-20261109` の履歴保存commitは [`054fa91493713fbf8d233828e9070dc07b8c2a1c`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/054fa91493713fbf8d233828e9070dc07b8c2a1c)。第一parentは最新REPORT `776e0a16ac98762631fe9307672725fdfee3b73d`、追加parentは独立した9系統の先端。tree SHAは双方とも`b2961b6f29e78a9c110627959a3f076fc51fc502`で、tree差分0件。

これは**履歴だけの保全**であり、9系統の科学コード・結果を最新版の作業treeへ取り込んだという意味ではない。元のSOURCE・結果・STOPは各parentの元commit/treeに残る。既存branchの再merge、科学的な統合の主張、競合解消、科学コード変更は行わない。保存tagは新設せず、既存103本を保持した。

| 保存した独立系統のparent | 内容 |
|---|---|
| `169e327ae50960173bb2690ab98ea92c85e7a5ed` | Integrate near-objective and common ideal research C evidence |
| `1a588f5491bce394ecf039ee4d2de57581e2e0b1` | docs(research-c): map prior art and prepare paper evidence and figure plan |
| `3718d369b3db515b2dd13b2cdbb6c49d57880121` | Validate update064 r32 price advantage with independent 256-sample holdout |
| `39ee9705b7a459c40c205c542d660b5d2ab90d33` | Record saved CDF8/XDF8 common ideal lane execution |
| `b7a3db09e395d36965e5b9dc28ae4a299db3b2f0` | Record completed 27-call B2 reference bound evaluation and mandatory STOP |
| `c0213176181b0f84eca0cf96464246d6a14e63a7` | Execute fixed seven-call native pair batch pilot |
| `c298af7ef06192a7d448507abba0ba3b587784e0` | Audit block cancellation loss and retain Pauli-frame corrections |
| `c67219fb6f639346ca8930f06832560b398c173f` | Record published four-call diagnosis verification |
| `f2a9baea08eeeeb2af7b4a87445acdc61932c911` | Document unused-condition transfer design and technical readiness |

過去の整理記録は、2026-11-02の`7ddf2bdf93dadb8c9868676636e06d6b9ca13143`にある`docs/research/branch_archive_20261102/{HANDOFF.md,README.md,branches.csv,manifest.json}`と、2026-10-27の`5ee9dcff7976e8c353792b8e79fa750d47c6ad82`にある`inventory/branches/github_branch_cleanup_review_20261027/`の分類・重要履歴台帳を確認した。11/2台帳の研究C 164行、旧分類87行・重要履歴157行を照合した。現存83本はこれら旧branch名と重複せず、旧保持判断を削除候補へ読み替えていない。

11/2の明示KEEPのうち`codex/research-c-pr-integrated-20261102`と`codex/research-c-pr-resource-contribution-20261102`は、着手時点でlocal/実remoteとも既に存在しなかった。本作業による削除ではない。`main`、対象外KEEP、既存archive tagは保持した。

既存`scripts/retire_archived_remote_branches.py`の`deletion_arguments`だけを再利用して、58本に対するatomic・expected SHA付きの削除引数を生成した。旧354本計画全体のrunnerは実行していない。

## SOURCE・結果・STOP・入力の照合

研究Cのtracked文書・SOURCE/契約/結果JSON・証拠索引の静的blob 2,068件を読み、文書中のcommit参照385件と旧保全台帳、指定9件を突き合わせ、計387 commitを保持対象として確認した。[preserved_commits.csv](research_c_branch_cleanup_20261109/preserved_commits.csv) はfull SHA、参照例、保持remote ref、その観測tip、祖先到達可能性を示す。参照先に他研究の既存資料も含むが、branch整理対象へは広げていない。

保持根拠は、削除失敗した58本自身を除外したref集合を用いた。各元HEADとSOURCE/結果参照は、新しい保存ref、worktree使用branchのremote、既存archive tagまたは対象外の保持refから到達可能。単なるreflog・未公開tag・手元のobject残存を根拠にしていない。

| 指定された固定点 | commit | 到達可能 |
|---|---|---|
| PR/ENTRY integrated result | `417fa459b823509be5c2010c5836e58955088282` | 保持remote refから確認済み |
| relocated P/ENTRY result | `3015d4fb342d4d98b58e41d57553e75539a07655` | 保持remote refから確認済み |
| CDF0 run02 result | `cbc517e75ad47054e4246748351f56269dac3a0c` | 保持remote refから確認済み |
| final synthesis document | `3eccab64a2672813e479c113c2485219f8479885` | 保持remote refから確認済み |
| P0 termination/Rust CASE D document | `ef6b8e8df53120667257ecd38c8f4813fdf9f852` | 保持remote refから確認済み |
| original three-method document | `709813805d41e470676e311cd63a740af23bf33c` | 保持remote refから確認済み |
| independent128 preparation | `2dba3fb09800e373ccb94027a2a9da1d2c540716` | 保持remote refから確認済み |
| independent128 result | `e09f3bea9df5f2f545f1b928e6cf164d4e5d4551` | 保持remote refから確認済み |
| latest three-method document | `776e0a16ac98762631fe9307672725fdfee3b73d` | 保持remote refから確認済み |

実行SOURCEと後日公開結果を区別した主要な対応は次のとおり。

| 系列 | 科学SOURCE / 実装 | 結果公開commit |
|---|---|---|
| 元辞書PR/ENTRY fit | 独立SOURCE commitは既存資料でも未特定。保存manifest SHA256 `200418968aa9e60e5ce842425e0373a4d8fafdb4f78c53a22df37b204989afd6`を維持 | `417fa459b823509be5c2010c5836e58955088282` |
| 再配置P/E継続run | `cfc64b70861a6482dddd042da8a110abebd6ad7a` / manifest `6bd88bc86bd1da2876f87b8ae67be509b15727f01517cea9c7219f91c350c4cc` | `3015d4fb342d4d98b58e41d57553e75539a07655` |
| CDF0 run02 | `bde646c2b0d275c66d9890eee835d4692f55cafb` / manifest `640052dfeed3b886bb95e11abb410e3275cdda1ffe82661e544037c120f94c32` | `cbc517e75ad47054e4246748351f56269dac3a0c` |
| 独立128×2 | `4f73bec3184523207777ce7f09c75c7cdfd3cb31` / manifest `934bfff3923609b5add4b9ec33ea2a9f2f3ac9004f41d7768f77251e62fb6f89` | `e09f3bea9df5f2f545f1b928e6cf164d4e5d4551` |
| Stage A | 公開SOURCE `e1085c95f5c01b68566e17d543ace5832cb94061`、数値実装`5a185964090f2cb8d19d445d41cf520c4d8ed2c7` | 元CDF0 fit・neutral起点の区別を維持 |

[protected_artifacts.csv](research_c_branch_cleanup_20261109/protected_artifacts.csv) に、SOURCE、固定結果/契約/STOP metadata、tracked入力・候補NPZ、公開raw packageの1,171個のblobを固定commit/pathとともに記録した。Git object IDと保持commitの祖先関係を照合しており、内容の書換えをしていない。静的抽出で得たSTOP/completion/marker blobは230個。旧STOPを成功結果で上書きしていない。

入力J/Uの候補識別はCDF0 `f3a3141059e29f79f39c3f6d3721562abfa7221ad97ce06a118331f6583e1a02`、P `1bd7c00ec11ef468993b36edcd267b84f921d66acb8994a4f6a9c751ddb66cfb`、E2 `0e1119dfb6be9b75fb5e592b7a65c76cda61e348fd7cad65d7f104bf997929f5`という既存bindingを維持。共通inputは`1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`。配列をdecodeして新たに数値照合したものではない。

raw保管は元報告の範囲を維持した。統合runの原サーバーworktree（`research-c-integrated-server-20261106` / `research-c-integrated-picklefix-20261106`等）や再配置runの保管ディレクトリは操作していない。独立128×2の全2882件の可逆公開packageは、`e09f...`の`artifacts/research_c_independent_price_precision/2026-11-08/publication/`以下で元Git blobのまま到達可能。未公開NPZ/QPYの存在・byteを今回新規監査したとは主張しない。既存保存場所・保全記録を変えず、rawの展開・再解析・複製・削除は行わない。

文書中に現れた本repoにない40桁文字列41種は、外部ライブラリcommit/blob40種と有理数の分子1種だった。本repo科学SOURCE欠落とは分類しない。外部SOURCE記録自体は元metadata blobのまま保全した。

## remote削除失敗と残作業

GitHub branch一覧405本は全件`protected=false`、ruleset一覧は空、open PR一覧は空であった。削除直前にもopen PRが空であること、58本の実remote HEADが棚卸し値と一致し、保存先remote refが公開SHAを指すことを再確認した。

2026-11-09 07:40:20 UTC、58本のatomic削除pushはexit 128で失敗した。各refに`--force-with-lease=refs/heads/<name>:<old SHA>`を付けた。これは削除対象SHAの一致を条件にするためで、履歴書換えのforce pushは行っていない。失敗後の`git ls-remote`で全58本の元SHAが残っていることを確認した。

```text
fatal: could not read Username for 'https://github.com': terminal prompts disabled
```

既存HTTPS credential/helperは利用できず、既存SSH鍵・known_hostsもなかった。SSHはstrict host key確認で停止した。GitHub connectorはcommit・branch作成・ref更新を提供するが、branch削除のtoolは提供しない。認証情報・credential helper・SSH鍵・Git設定は変更していない。

失敗した58本（元HEAD・事前分類はCSVにも記録）:

| Branch | 元HEAD | 事前分類 |
|---|---|---|
| `codex/research-c-all-law-signal-ca-readiness-20261103` | `3a784888a188db946681a6143c5e18479ddf284a` | PRESERVE_THEN_DELETE |
| `codex/research-c-b1-b2-accuracy-scope-integration-20261103` | `90a2a7baf07b98108111126504d202294a884af5` | PRESERVE_THEN_DELETE |
| `codex/research-c-b1-b2-reference-bound-integration-20261103` | `891aa78f47e66184eaf8a6d527ca8773ba6bbdd0` | PRESERVE_THEN_DELETE |
| `codex/research-c-block-witness-cancellation-readiness-20261105` | `c298af7ef06192a7d448507abba0ba3b587784e0` | PRESERVE_THEN_DELETE |
| `codex/research-c-cdf0-entry-independent-price-precision-readiness-20261108` | `e09f3bea9df5f2f545f1b928e6cf164d4e5d4551` | DELETE_MERGED |
| `codex/research-c-cdf0-z-relocation-comparison-execution-20261108` | `4ef705940b83d7c60de5966611f2d265435336f8` | DELETE_MERGED |
| `codex/research-c-cdf0-z-relocation-comparison-preparation-20261108` | `bde646c2b0d275c66d9890eee835d4692f55cafb` | DELETE_MERGED |
| `codex/research-c-cdf0-z-relocation-comparison-rerun-20261108` | `cbc517e75ad47054e4246748351f56269dac3a0c` | DELETE_MERGED |
| `codex/research-c-compiled-raw-certificate-acceleration-20261105` | `6bcf68d55ddeab852b275940fcdc78bccceba04f` | PRESERVE_THEN_DELETE |
| `codex/research-c-compiled-raw-local-pilot-execution-20261105` | `f6b1516116d40c3d8d1467e476abb032a0f562b6` | PRESERVE_THEN_DELETE |
| `codex/research-c-cx-structural-block-pilot-execution-20261105` | `acdb1c404b21c004c16d55a702446b02fe67f2a6` | PRESERVE_THEN_DELETE |
| `codex/research-c-cx-structural-block-readiness-20261105` | `7f70ca0af877621d6e4abb5db94945819ee0826e` | PRESERVE_THEN_DELETE |
| `codex/research-c-direct-m0-bound-execution-20261105` | `75fe82e9d627bfb0ceca450b96459144e6627914` | PRESERVE_THEN_DELETE |
| `codex/research-c-direct-m0-oracle-binding-20261105` | `2c0f52632316be8bbf39d9e40b138d0fd669f860` | PRESERVE_THEN_DELETE |
| `codex/research-c-direct-round-m0-runner-preparation-20261105` | `e409111ccd1427c343c2c44a4e87a3230f70a824` | PRESERVE_THEN_DELETE |
| `codex/research-c-direct-round-rigorous-signal-readiness-20261105` | `db6313156f97a37c2710f56ed7a93a34967d58f4` | PRESERVE_THEN_DELETE |
| `codex/research-c-entry-e2-r16-all-law-readiness-20261108` | `10269db6b4ad27707dcfe624dd487508bab733a8` | DELETE_MERGED |
| `codex/research-c-entry-e2-r16-p0-termination-rust-uniform-20261108` | `ef6b8e8df53120667257ecd38c8f4813fdf9f852` | DELETE_MERGED |
| `codex/research-c-entry-e2-r16-p0-uniform-feasibility-20261108` | `95a828cf2ad099e8ff7a6bbe470be1fc94cfb79b` | DELETE_MERGED |
| `codex/research-c-fixed-pair-batch-pilot-execution-20261104` | `c0213176181b0f84eca0cf96464246d6a14e63a7` | PRESERVE_THEN_DELETE |
| `codex/research-c-fixed-word-pilot-execution-20261104` | `3649b75af191916daf44abb91e14d34c43b83e8f` | PRESERVE_THEN_DELETE |
| `codex/research-c-fixed-word-pilot-preparation-20261104` | `ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7` | PRESERVE_THEN_DELETE |
| `codex/research-c-fresh-iid-calibration-infrastructure-20261104` | `9098de49f1bd816dbaf810eb63815c841169236d` | PRESERVE_THEN_DELETE |
| `codex/research-c-h4-relocation-preparation-20261106` | `052941ab7ce680f57b0b5ba9ef27ef8cdd4937c1` | DELETE_MERGED |
| `codex/research-c-head-bridge-execution-20261104` | `369e2dfad9a4251ba5eb2fda28f183d559a791cc` | PRESERVE_THEN_DELETE |
| `codex/research-c-head-bridge-stage-p-readiness-20261104` | `f658f72d5f309538cba23b0745c7e9389411758f` | PRESERVE_THEN_DELETE |
| `codex/research-c-hermitian-oracle-bridge-readiness-20261105` | `56d2f85e471d2ab3927c9c9568f31034df143b26` | PRESERVE_THEN_DELETE |
| `codex/research-c-iid-calibration-integration-20261104` | `325f07ae02d8872d4fe322d6ad6e08c256e0bef7` | PRESERVE_THEN_DELETE |
| `codex/research-c-iid-grid-all-law-readiness-20261103` | `ae9a4e13d1ab6f48fc4d38cbbcd0fddca30e9c53` | PRESERVE_THEN_DELETE |
| `codex/research-c-iid-grid-expected-bias-readiness-20261104` | `bd560a45410409573ae849ca5aaf4050343910bf` | PRESERVE_THEN_DELETE |
| `codex/research-c-integrated-execution-20261105` | `a7face80e10fd74e1a49dc530409db5b7f677aaa` | DELETE_MERGED |
| `codex/research-c-latest-evidence-integration-20261103` | `ba0f280c8be9e7038c5fa6b5fd390a235b113833` | DELETE_MERGED |
| `codex/research-c-near-objective-practical-comparison-preparation-20261103` | `70a98adfb701a209db2c65ea60fb789ff62b1fe7` | DELETE_MERGED |
| `codex/research-c-near-objective-practical-execution-20261103` | `2ceb0a8825785ebd4f21241d8ef0f20cb321bf69` | DELETE_MERGED |
| `codex/research-c-near-objective-transfer-readiness-20261103` | `f2a9baea08eeeeb2af7b4a87445acdc61932c911` | PRESERVE_THEN_DELETE |
| `codex/research-c-near-objective-xdf-evidence-integration-20261103` | `169e327ae50960173bb2690ab98ea92c85e7a5ed` | PRESERVE_THEN_DELETE |
| `codex/research-c-near-practical-r32-downstream-execution-20261103` | `beb8de5cd8b71f80bb69556f92299435ff431c3e` | DELETE_MERGED |
| `codex/research-c-p0-compiler-totality-readiness-20261104` | `60dcaa94fd44675bc3184e409aa4ebdef7a07dbf` | PRESERVE_THEN_DELETE |
| `codex/research-c-p0-uniform-method-readiness-20261103` | `69c9a764aee66cf2ca79ab29bd3145a027385bcc` | PRESERVE_THEN_DELETE |
| `codex/research-c-per-word-population-range-readiness-20261104` | `6c1fbe5399b2418e680acaa4f6e8aaa5e27d6dbd` | PRESERVE_THEN_DELETE |
| `codex/research-c-population-range-stage-R-execution-20261104` | `e73f79de6e46d65e76ac7126f4eaf8a9b3568c0b` | PRESERVE_THEN_DELETE |
| `codex/research-c-pr-xdf-cdf-statistical-comparison-preparation-20261103` | `2f1c86f161fb76e84dfa6b31f6912eb8a01ccb8f` | PRESERVE_THEN_DELETE |
| `codex/research-c-reference-bound-contract-readiness-20261103` | `b7a3db09e395d36965e5b9dc28ae4a299db3b2f0` | PRESERVE_THEN_DELETE |
| `codex/research-c-reference-numerical-uncertainty-readiness-20261103` | `2ac4a38dced97406d15fcd157e9d7a0ec7aba982` | PRESERVE_THEN_DELETE |
| `codex/research-c-related-work-paper-readiness-20261105` | `1a588f5491bce394ecf039ee4d2de57581e2e0b1` | PRESERVE_THEN_DELETE |
| `codex/research-c-robustness-retuning-transfer-preparation-20261105` | `9ff0ef7449c5a53e559be837647937cf1d011391` | DELETE_MERGED |
| `codex/research-c-saved-eigh-difference-execution-20261105` | `c67219fb6f639346ca8930f06832560b398c173f` | PRESERVE_THEN_DELETE |
| `codex/research-c-saved-eigh-difference-readiness-20261105` | `b7ebd03c087f986abdf24abd8be4303a28f02308` | PRESERVE_THEN_DELETE |
| `codex/research-c-stage-R-schema-execution-20261104` | `1c0e1d1b4967efecb08dd3d8e7a747eba6c83039` | PRESERVE_THEN_DELETE |
| `codex/research-c-stage-R-schema-repair-20261104` | `60cdba8fe475acb728e86e7733694ac8211d2ec7` | PRESERVE_THEN_DELETE |
| `codex/research-c-stage-p-completion-20261104` | `c01bd2742ca88071d17db1fc3e3aec6cb9f3cb9e` | PRESERVE_THEN_DELETE |
| `codex/research-c-stage-p-execution-20261104` | `5e1450b1071359b81819fe5de481cdeccc9d3f35` | PRESERVE_THEN_DELETE |
| `codex/research-c-stage-p-totality-calibration-feasibility-20261104` | `ce3185513578a8329efe364c748b5ce2aacdda45` | PRESERVE_THEN_DELETE |
| `codex/research-c-strong-xdf-cdf-comparison-audit-20261103` | `8375433869999c2e10501b1983c11b2b22c5d423` | PRESERVE_THEN_DELETE |
| `codex/research-c-update064-r32-price-statistics-execution-20261103` | `3718d369b3db515b2dd13b2cdbb6c49d57880121` | PRESERVE_THEN_DELETE |
| `codex/research-c-uwc-rcdf-near-objective-comparison-design-20261103` | `2459e018ea2276c359d6fe96253fbcad0bfd65bb` | DELETE_MERGED |
| `codex/research-c-xdf-cdf-common-ideal-lane-execution-20261103` | `39ee9705b7a459c40c205c542d660b5d2ab90d33` | PRESERVE_THEN_DELETE |
| `codex/research-c-xdf-cdf-common-ideal-lane-preparation-20261103` | `2d20ad196a406752de8e4c0ce9516b101e49d927` | PRESERVE_THEN_DELETE |

次の担当は**ユーザー（既存認証を利用できる環境から、上記58本の削除を実行）**。実行前にworktree・PR・保持ref・HEADを再確認し、新規使用やHEAD変化があれば該当branchを削除しない。今回の実行許可はこの整理範囲に対して既に得ているが、認証設定の変更を自動で行う許可とは扱わない。

失敗したpush全体の再実行コマンド（実行時に上記の状態を再確認すること）:

```bash
GIT_TERMINAL_PROMPT=0 git -C /home/AbeHiromu/projects/partially-randomized-trotter push --atomic --force-with-lease=refs/heads/codex/research-c-all-law-signal-ca-readiness-20261103:3a784888a188db946681a6143c5e18479ddf284a --force-with-lease=refs/heads/codex/research-c-b1-b2-accuracy-scope-integration-20261103:90a2a7baf07b98108111126504d202294a884af5 --force-with-lease=refs/heads/codex/research-c-b1-b2-reference-bound-integration-20261103:891aa78f47e66184eaf8a6d527ca8773ba6bbdd0 --force-with-lease=refs/heads/codex/research-c-block-witness-cancellation-readiness-20261105:c298af7ef06192a7d448507abba0ba3b587784e0 --force-with-lease=refs/heads/codex/research-c-cdf0-entry-independent-price-precision-readiness-20261108:e09f3bea9df5f2f545f1b928e6cf164d4e5d4551 --force-with-lease=refs/heads/codex/research-c-cdf0-z-relocation-comparison-execution-20261108:4ef705940b83d7c60de5966611f2d265435336f8 --force-with-lease=refs/heads/codex/research-c-cdf0-z-relocation-comparison-preparation-20261108:bde646c2b0d275c66d9890eee835d4692f55cafb --force-with-lease=refs/heads/codex/research-c-cdf0-z-relocation-comparison-rerun-20261108:cbc517e75ad47054e4246748351f56269dac3a0c --force-with-lease=refs/heads/codex/research-c-compiled-raw-certificate-acceleration-20261105:6bcf68d55ddeab852b275940fcdc78bccceba04f --force-with-lease=refs/heads/codex/research-c-compiled-raw-local-pilot-execution-20261105:f6b1516116d40c3d8d1467e476abb032a0f562b6 --force-with-lease=refs/heads/codex/research-c-cx-structural-block-pilot-execution-20261105:acdb1c404b21c004c16d55a702446b02fe67f2a6 --force-with-lease=refs/heads/codex/research-c-cx-structural-block-readiness-20261105:7f70ca0af877621d6e4abb5db94945819ee0826e --force-with-lease=refs/heads/codex/research-c-direct-m0-bound-execution-20261105:75fe82e9d627bfb0ceca450b96459144e6627914 --force-with-lease=refs/heads/codex/research-c-direct-m0-oracle-binding-20261105:2c0f52632316be8bbf39d9e40b138d0fd669f860 --force-with-lease=refs/heads/codex/research-c-direct-round-m0-runner-preparation-20261105:e409111ccd1427c343c2c44a4e87a3230f70a824 --force-with-lease=refs/heads/codex/research-c-direct-round-rigorous-signal-readiness-20261105:db6313156f97a37c2710f56ed7a93a34967d58f4 --force-with-lease=refs/heads/codex/research-c-entry-e2-r16-all-law-readiness-20261108:10269db6b4ad27707dcfe624dd487508bab733a8 --force-with-lease=refs/heads/codex/research-c-entry-e2-r16-p0-termination-rust-uniform-20261108:ef6b8e8df53120667257ecd38c8f4813fdf9f852 --force-with-lease=refs/heads/codex/research-c-entry-e2-r16-p0-uniform-feasibility-20261108:95a828cf2ad099e8ff7a6bbe470be1fc94cfb79b --force-with-lease=refs/heads/codex/research-c-fixed-pair-batch-pilot-execution-20261104:c0213176181b0f84eca0cf96464246d6a14e63a7 --force-with-lease=refs/heads/codex/research-c-fixed-word-pilot-execution-20261104:3649b75af191916daf44abb91e14d34c43b83e8f --force-with-lease=refs/heads/codex/research-c-fixed-word-pilot-preparation-20261104:ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7 --force-with-lease=refs/heads/codex/research-c-fresh-iid-calibration-infrastructure-20261104:9098de49f1bd816dbaf810eb63815c841169236d --force-with-lease=refs/heads/codex/research-c-h4-relocation-preparation-20261106:052941ab7ce680f57b0b5ba9ef27ef8cdd4937c1 --force-with-lease=refs/heads/codex/research-c-head-bridge-execution-20261104:369e2dfad9a4251ba5eb2fda28f183d559a791cc --force-with-lease=refs/heads/codex/research-c-head-bridge-stage-p-readiness-20261104:f658f72d5f309538cba23b0745c7e9389411758f --force-with-lease=refs/heads/codex/research-c-hermitian-oracle-bridge-readiness-20261105:56d2f85e471d2ab3927c9c9568f31034df143b26 --force-with-lease=refs/heads/codex/research-c-iid-calibration-integration-20261104:325f07ae02d8872d4fe322d6ad6e08c256e0bef7 --force-with-lease=refs/heads/codex/research-c-iid-grid-all-law-readiness-20261103:ae9a4e13d1ab6f48fc4d38cbbcd0fddca30e9c53 --force-with-lease=refs/heads/codex/research-c-iid-grid-expected-bias-readiness-20261104:bd560a45410409573ae849ca5aaf4050343910bf --force-with-lease=refs/heads/codex/research-c-integrated-execution-20261105:a7face80e10fd74e1a49dc530409db5b7f677aaa --force-with-lease=refs/heads/codex/research-c-latest-evidence-integration-20261103:ba0f280c8be9e7038c5fa6b5fd390a235b113833 --force-with-lease=refs/heads/codex/research-c-near-objective-practical-comparison-preparation-20261103:70a98adfb701a209db2c65ea60fb789ff62b1fe7 --force-with-lease=refs/heads/codex/research-c-near-objective-practical-execution-20261103:2ceb0a8825785ebd4f21241d8ef0f20cb321bf69 --force-with-lease=refs/heads/codex/research-c-near-objective-transfer-readiness-20261103:f2a9baea08eeeeb2af7b4a87445acdc61932c911 --force-with-lease=refs/heads/codex/research-c-near-objective-xdf-evidence-integration-20261103:169e327ae50960173bb2690ab98ea92c85e7a5ed --force-with-lease=refs/heads/codex/research-c-near-practical-r32-downstream-execution-20261103:beb8de5cd8b71f80bb69556f92299435ff431c3e --force-with-lease=refs/heads/codex/research-c-p0-compiler-totality-readiness-20261104:60dcaa94fd44675bc3184e409aa4ebdef7a07dbf --force-with-lease=refs/heads/codex/research-c-p0-uniform-method-readiness-20261103:69c9a764aee66cf2ca79ab29bd3145a027385bcc --force-with-lease=refs/heads/codex/research-c-per-word-population-range-readiness-20261104:6c1fbe5399b2418e680acaa4f6e8aaa5e27d6dbd --force-with-lease=refs/heads/codex/research-c-population-range-stage-R-execution-20261104:e73f79de6e46d65e76ac7126f4eaf8a9b3568c0b --force-with-lease=refs/heads/codex/research-c-pr-xdf-cdf-statistical-comparison-preparation-20261103:2f1c86f161fb76e84dfa6b31f6912eb8a01ccb8f --force-with-lease=refs/heads/codex/research-c-reference-bound-contract-readiness-20261103:b7a3db09e395d36965e5b9dc28ae4a299db3b2f0 --force-with-lease=refs/heads/codex/research-c-reference-numerical-uncertainty-readiness-20261103:2ac4a38dced97406d15fcd157e9d7a0ec7aba982 --force-with-lease=refs/heads/codex/research-c-related-work-paper-readiness-20261105:1a588f5491bce394ecf039ee4d2de57581e2e0b1 --force-with-lease=refs/heads/codex/research-c-robustness-retuning-transfer-preparation-20261105:9ff0ef7449c5a53e559be837647937cf1d011391 --force-with-lease=refs/heads/codex/research-c-saved-eigh-difference-execution-20261105:c67219fb6f639346ca8930f06832560b398c173f --force-with-lease=refs/heads/codex/research-c-saved-eigh-difference-readiness-20261105:b7ebd03c087f986abdf24abd8be4303a28f02308 --force-with-lease=refs/heads/codex/research-c-stage-R-schema-execution-20261104:1c0e1d1b4967efecb08dd3d8e7a747eba6c83039 --force-with-lease=refs/heads/codex/research-c-stage-R-schema-repair-20261104:60cdba8fe475acb728e86e7733694ac8211d2ec7 --force-with-lease=refs/heads/codex/research-c-stage-p-completion-20261104:c01bd2742ca88071d17db1fc3e3aec6cb9f3cb9e --force-with-lease=refs/heads/codex/research-c-stage-p-execution-20261104:5e1450b1071359b81819fe5de481cdeccc9d3f35 --force-with-lease=refs/heads/codex/research-c-stage-p-totality-calibration-feasibility-20261104:ce3185513578a8329efe364c748b5ce2aacdda45 --force-with-lease=refs/heads/codex/research-c-strong-xdf-cdf-comparison-audit-20261103:8375433869999c2e10501b1983c11b2b22c5d423 --force-with-lease=refs/heads/codex/research-c-update064-r32-price-statistics-execution-20261103:3718d369b3db515b2dd13b2cdbb6c49d57880121 --force-with-lease=refs/heads/codex/research-c-uwc-rcdf-near-objective-comparison-design-20261103:2459e018ea2276c359d6fe96253fbcad0bfd65bb --force-with-lease=refs/heads/codex/research-c-xdf-cdf-common-ideal-lane-execution-20261103:39ee9705b7a459c40c205c542d660b5d2ab90d33 --force-with-lease=refs/heads/codex/research-c-xdf-cdf-common-ideal-lane-preparation-20261103:2d20ad196a406752de8e4c0ce9516b101e49d927 origin :refs/heads/codex/research-c-all-law-signal-ca-readiness-20261103 :refs/heads/codex/research-c-b1-b2-accuracy-scope-integration-20261103 :refs/heads/codex/research-c-b1-b2-reference-bound-integration-20261103 :refs/heads/codex/research-c-block-witness-cancellation-readiness-20261105 :refs/heads/codex/research-c-cdf0-entry-independent-price-precision-readiness-20261108 :refs/heads/codex/research-c-cdf0-z-relocation-comparison-execution-20261108 :refs/heads/codex/research-c-cdf0-z-relocation-comparison-preparation-20261108 :refs/heads/codex/research-c-cdf0-z-relocation-comparison-rerun-20261108 :refs/heads/codex/research-c-compiled-raw-certificate-acceleration-20261105 :refs/heads/codex/research-c-compiled-raw-local-pilot-execution-20261105 :refs/heads/codex/research-c-cx-structural-block-pilot-execution-20261105 :refs/heads/codex/research-c-cx-structural-block-readiness-20261105 :refs/heads/codex/research-c-direct-m0-bound-execution-20261105 :refs/heads/codex/research-c-direct-m0-oracle-binding-20261105 :refs/heads/codex/research-c-direct-round-m0-runner-preparation-20261105 :refs/heads/codex/research-c-direct-round-rigorous-signal-readiness-20261105 :refs/heads/codex/research-c-entry-e2-r16-all-law-readiness-20261108 :refs/heads/codex/research-c-entry-e2-r16-p0-termination-rust-uniform-20261108 :refs/heads/codex/research-c-entry-e2-r16-p0-uniform-feasibility-20261108 :refs/heads/codex/research-c-fixed-pair-batch-pilot-execution-20261104 :refs/heads/codex/research-c-fixed-word-pilot-execution-20261104 :refs/heads/codex/research-c-fixed-word-pilot-preparation-20261104 :refs/heads/codex/research-c-fresh-iid-calibration-infrastructure-20261104 :refs/heads/codex/research-c-h4-relocation-preparation-20261106 :refs/heads/codex/research-c-head-bridge-execution-20261104 :refs/heads/codex/research-c-head-bridge-stage-p-readiness-20261104 :refs/heads/codex/research-c-hermitian-oracle-bridge-readiness-20261105 :refs/heads/codex/research-c-iid-calibration-integration-20261104 :refs/heads/codex/research-c-iid-grid-all-law-readiness-20261103 :refs/heads/codex/research-c-iid-grid-expected-bias-readiness-20261104 :refs/heads/codex/research-c-integrated-execution-20261105 :refs/heads/codex/research-c-latest-evidence-integration-20261103 :refs/heads/codex/research-c-near-objective-practical-comparison-preparation-20261103 :refs/heads/codex/research-c-near-objective-practical-execution-20261103 :refs/heads/codex/research-c-near-objective-transfer-readiness-20261103 :refs/heads/codex/research-c-near-objective-xdf-evidence-integration-20261103 :refs/heads/codex/research-c-near-practical-r32-downstream-execution-20261103 :refs/heads/codex/research-c-p0-compiler-totality-readiness-20261104 :refs/heads/codex/research-c-p0-uniform-method-readiness-20261103 :refs/heads/codex/research-c-per-word-population-range-readiness-20261104 :refs/heads/codex/research-c-population-range-stage-R-execution-20261104 :refs/heads/codex/research-c-pr-xdf-cdf-statistical-comparison-preparation-20261103 :refs/heads/codex/research-c-reference-bound-contract-readiness-20261103 :refs/heads/codex/research-c-reference-numerical-uncertainty-readiness-20261103 :refs/heads/codex/research-c-related-work-paper-readiness-20261105 :refs/heads/codex/research-c-robustness-retuning-transfer-preparation-20261105 :refs/heads/codex/research-c-saved-eigh-difference-execution-20261105 :refs/heads/codex/research-c-saved-eigh-difference-readiness-20261105 :refs/heads/codex/research-c-stage-R-schema-execution-20261104 :refs/heads/codex/research-c-stage-R-schema-repair-20261104 :refs/heads/codex/research-c-stage-p-completion-20261104 :refs/heads/codex/research-c-stage-p-execution-20261104 :refs/heads/codex/research-c-stage-p-totality-calibration-feasibility-20261104 :refs/heads/codex/research-c-strong-xdf-cdf-comparison-audit-20261103 :refs/heads/codex/research-c-update064-r32-price-statistics-execution-20261103 :refs/heads/codex/research-c-uwc-rcdf-near-objective-comparison-design-20261103 :refs/heads/codex/research-c-xdf-cdf-common-ideal-lane-execution-20261103 :refs/heads/codex/research-c-xdf-cdf-common-ideal-lane-preparation-20261103
```

## 削除後監査とSTOP

- 元83本すべてのHEADは、削除候補自身を除いたremote保持refから到達可能。固定commit 387件・重要blob 1171件を同様に照合した。
- local削除1本の不存在、remote削除0本・失敗58本の元SHA残存を確認。KEEPとworktree使用branchは残る。
- 既存local/remote HEADの移動なし、remote tag103本のSHA変更なし。対象外の新規branch追加は観測のみ。既存worktree・process・共有設定は操作しない。
- 最新REPORTの元commit・元tree・図と科学SOURCE・結果・STOPは改変しない。本公開commitで追加するのは本報告と3つのCSVだけ。
- GitHubへの報告公開・tree差分照合は完了チャットで固定commit SHAを報告する。

正式共通運用指示`research_project_instructions_unified_2026-10-25.md`とプロジェクトAGENTS.mdを適用。主エージェント自身によるGit技術照合であり、別レビュアーによる独立科学レビューではない。

**MANDATORY STOP。Codexは追加の科学計算・再解析・重要レビュー・認証設定変更へ進まない。残作業は認証可能な環境でのremote58本削除のみ。**


## 追記：認証可能な環境でのremote削除・事後監査（2026-11-09 17:18 JST）

本節より上は整理commit `8d174496761ac3ae91892c11ad83bf236d1b050a` の過去時点の記録をそのまま保持する。本節は、その時点で認証失敗となった58本だけを対象にした後続実行の記録である。新たな削除候補は追加していない。

**56本のremote削除を完了。残る2本は現在の環境で実在worktreeに使用されているため、追加HOLDとした。** 科学計算・数値再解析・DF fit・統計解析・回路build/compile・重要GPT科学レビューはすべて0件。正式共通運用指示 `research_project_instructions_unified_2026-10-25.md` と実際のAGENTS.mdを確認・適用した。

| 項目 | 今回の結果 |
|---|---:|
| 当初削除対象／削除直前にremoteに実在 | 58本／58本 |
| 今回削除成功 | 56本 |
| ALREADY_ABSENT | 0本 |
| remote HEAD変更によるHOLD | 0本 |
| local HEAD変更によるHOLD | 0本 |
| 保護・ruleset・open PRによるHOLD | 0本 |
| worktreeによる追加HOLD | 2本 |
| その他失敗 | 0本 |
| 今回の対象でremoteに残存 | 2本 |
| 研究C remote | 81本 → 25本 |
| 全remote branch | 410本 → 354本 |
| remote tag | 103本 → 103本（全SHA不変） |

各対象の元full SHA、削除直前・直後のremote SHA、local SHA、worktree観測、最終結果は [今回の58本の結果CSV](research_c_branch_cleanup_20261109/remote_deletion_results_20261109.csv) に記録した。前回の `branches.csv`・`preserved_commits.csv`・`protected_artifacts.csv` は変更していない。

### 最新状態と追加HOLD

既存のSSH認証をそのまま利用し、対象remoteが `git@github.com:HIROMU1015/Partially-Randomized-Trotter.git` であることを確認した。全58本のremote HEADは前回記録SHAと一致し、現環境のlocal HEADにも相違はなかった。GitHub APIのprotected branch一覧、親由来を含むruleset一覧、open PR一覧はいずれも空だった。認証情報・SSH鍵・credential helper・共有Git設定は変更していない。

| 追加HOLD branch | 元HEAD・今回のremote HEAD | 現在のworktree |
|---|---|---|
| `codex/research-c-integrated-execution-20261105` | `a7face80e10fd74e1a49dc530409db5b7f677aaa` | `/home/abe/Project/Partially Randomized Trotter/.worktrees/research-c-integrated-preparation-20261105` |
| `codex/research-c-related-work-paper-readiness-20261105` | `1a588f5491bce394ecf039ee4d2de57581e2e0b1` | `/home/abe/Project/Partially Randomized Trotter/.worktrees/research-c-related-work-paper-readiness-20261105` |

対象58本のうち47本には現環境のworktree登録があった。上記2本はpathと`.git`が実在する。残る45本はGit自身が `prunable: gitdir file points to non-existent location` と報告し、path・`.git`とも存在せず、lockもない過去の登録であり、現在のworktree使用とは扱わなかった。これらの登録・local branch・reflogを消す操作は行っていない。

前回HOLD 24本は今回も対象外。**そのうちremoteに存在するのは当初から21本**で、全21本の名前・SHAは削除前後で同一だった。残る次の3本は前回CSVの `remote_before=False`、`ABSENT_BEFORE` に対応し、今回もremoteに存在しない。新たにbranchを作成・復元する操作は行っていない。

- `codex/research-c-h4-relocated-downstream-execution-20261106`
- `codex/research-c-integrated-cgfix-20261106`
- `codex/research-c-integrated-server-20261106`

以前の実行環境のworktreeを新たに検査したとは主張しない。その環境への操作は0件。現環境では、全worktree登録と共有repositoryの全local branch refが作業前後で同一であることを確認した。checkout変更・worktree削除・移動・prune・GC・共有設定変更は行っていない。

### 削除実行とGitHub実refの事後照合

前回報告の58本分のlease引数をCSVと突き合わせ、追加HOLD 2本を除く56本だけを抽出した。2026-11-09 **08:18:35 UTC（17:18:35 JST）** に、各refへ記録済みexpected SHAの `--force-with-lease=refs/heads/<branch>:<SHA>` を指定した `git push --atomic` で削除した。終了コードは0。履歴上書き用の `git push --force` は使用していない。

[pushの各ref応答](research_c_branch_cleanup_20261109/remote_deletion_push_20261109.txt)、[削除直前の実remote ref](research_c_branch_cleanup_20261109/remote_refs_before_deletion_20261109.tsv)、[削除直後の実remote ref](research_c_branch_cleanup_20261109/remote_refs_after_deletion_20261109.tsv) を保存した。snapshotは `git ls-remote --heads --tags origin` がGitHubから返した実refであり、local remote-tracking refの状態を削除成功の根拠にしていない。

削除成功56本の不存在、KEEP 2本の存在、追加HOLD 2本の元SHAでの残存、既存remote HOLD 21本のSHA不変を確認した。削除した56本を除いたremote ref辞書は前後で完全一致した。したがって、削除操作による `main`、他研究、archive branch/tag、その他対象外branchの変更は0件。既存tagはannotated tagのpeeled refを含めて一致した。研究Cの25本はKEEP 2本＋以前からのremote HOLD 21本＋追加HOLD 2本である。

### 387 commit・1,171 blobとSOURCE・結果・STOPの保全

削除前と削除後の両方で、**今回の候補58本すべてを保持ref集合から除外**し、実remote refで観測した40本の保持ref（KEEP、従来HOLD、既存archive tag、既存の対象外保持ref）を根にGitグラフを照合した。削除したbranch自身や追加HOLD 2本のref、手元のreflogだけへの依存はない。

- `preserved_commits.csv` の387 commit：各記録の保持refの現remote tipから祖先として到達可能、**387/387 PASS**。
- `protected_artifacts.csv` の1,171 blob：記録されたcommitが現remote保持refの祖先であり、`commit:path`のGit object IDと種別が元のblobと一致し、保持tipからのobject到達可能性も確認、**1,171/1,171 PASS**。
- 元83本の全HEAD：削除候補を除外した保持refから到達可能、**83/83 PASS**。
- 履歴保存commit `054fa91493713fbf8d233828e9070dc07b8c2a1c` と最新REPORT保全commit `776e0a16ac98762631fe9307672725fdfee3b73d`：KEEP整理refから到達可能、**PASS**。9系統の独立履歴を含む祖先は変更していない。

対象blobの内訳はSOURCE 202、STOP 212、固定metadata/result/contract 604、入力・候補NPZまたは可逆公開raw package 153。主要なPR/ENTRY、Z再配置後、通常CDF0、独立128、三者比較REPORT、P0 CASE Dの固定commit・SOURCE・結果・STOP、および公開raw packageを含む。元のcommit/path/blobによる技術照合であり、配列decode・数値再解析・科学的内容の再評価は行っていない。既存の未公開raw保管場所を新たに検査したとは主張しない。

[実行・保全監査JSON](research_c_branch_cleanup_20261109/remote_execution_audit_20261109.json) に、実行時刻、集計、API観測、保持refとremote tip、削除前後の照合結果、元CSVのblob ID、前回HOLD全24本の今回のremote状態を記録した。既存コードの全体削除runnerは実行せず、前回の固定leaseコマンド・CSVとGit標準の到達可能性・path/object照合を再利用した。管理記録だけを分離した一時Git repositoryで作成し、既存worktree・dirty/untracked・科学SOURCE・入力・結果・rawは変更していない。

### 公開と停止点

本追記と5つの補助記録だけを、既存 `codex/research-c-branch-cleanup-20261109` の直系子commitとして公開する。before/after snapshotの整理ref SHAは本追記公開前の `8d174496761ac3ae91892c11ad83bf236d1b050a` であり、その後の公開操作は同整理refだけをfast-forwardする。公開後にGitHubの実branch SHA・commit tree・報告本文・補助記録を照合し、最終チャットで固定公開SHAを報告する。

**今回の58本は削除56本・使用中HOLD 2本として全件確定。MANDATORY STOP。** Codexは追加の削除・科学計算・再解析・重要レビューを開始しない。次の対応が必要な場合は、ユーザーがHOLD 2本のworktree利用を継続するか判断する。将来の解放・削除は本作業では実施せず、別途明示指示の対象とする。


## 別依頼の追補：全remote残存枝の棚卸し（2026-11-09）

[全remoteの最新分類・未確認事項・次の承認範囲](research_c_branch_cleanup_20261109/remaining_remote_audit_20261109/REPORT.md)。開始317本、別作業によるTrack C/O3の21本削除と使用中DF枝1本の公開を反映して297本。KEEP131・HOLD166・確定追加統合候補0・削除候補0。本監査による統合・削除・科学実行0。研究Cの従来削除56本とHOLD23本は変更せず、公開照合後mandatory STOP。
