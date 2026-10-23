# 既存環境・codeの再利用

venv作成前に、アクセス可能な自己の研究領域、pyenv versions、既存venv、wheel/cache、requirements/pyproject、過去の環境manifest・preflightを調べた。範囲を列挙したprivate inventoryを保全し、他ユーザー領域を探索していない。

14個のvenv設定ではPython3.10.12／3.11.0／3.11.1／3.12.3が見つかった。pyenvにも3.11.1と3.12.3が存在した。NumPy1.26.4の既存installationは10個でmetadataを確認できたが、Python3.11.9の適合専用環境は列挙範囲で見つからなかった。全hostに存在しないという主張ではない。既存venvは無変更。

自己のpip cacheにCPython3.11/Linux x86_64の該当NumPy1.26.4 wheelは取得できなかった。既存wheel候補4本はpip/setuptools用で、Python patch適合の根拠にはならない。そこでPyPI公開metadataを取得し、指定wheelの実bytesとSHA256を確認して専用wheelhouseへ保存した。

| 再利用候補 | 固定出所・適合性 | 今回の扱い |
|---|---|---|
| python-build | pyenv/pyenv `6393a4dfce90615792751c6567c07a118a961ff9`、3.11.9 definitionの公式archive SHA pin | 既存installerを変更せず呼出し。build/cache/temp/prefixを今回専用領域へ指定。shared pyenv/shimsを更新しない |
| P1* environment helper | [固定SOURCE](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/a80596c29e0eaaf7483f8981ed0e61c918335054/research_tools/p1star_foundation/environment.py) | `file_identity`を**逐字再利用**しAST/source segment一致を確認。RECORD inventory／実mapped libraryの確認方法も再利用 |
| 同helper全体 | z3/sympy/mpmath・z3 executableに固定され、実行すると今回対象外のpackageをimport | 全体import／validator呼出しはしない。NumPyとB専用identityへの最小管理wrapperを新設 |
| B-PRE contract.py | `3de42d3a3efe9ebc74dde8a1d0e1f1bdbadd607c`、stdlib gate／SOURCE・環境hash／schema／承認・one-shot | SOURCEを読むだけ。科学packageからimportせず、authorizationが必要なverify/consumeも呼ばない。既存gateを将来実行でそのまま使う |
| B-PRE管理audit | 既存のbytes/SHA256/Git blob、AST、固定Git参照、文書リンク照合 | 同じGit object/serializer規則を再利用。旧auditは旧namespaceへのwriteがあるためそのまま起動しない |
| H4 native receipt/preflight | `trottertracks`内の旧H4固有契約・科学runner・input依存 | 静的に確認。今回の研究SOURCE import禁止・science0・grant0と不適合なので起動しない |
| requirements.txt／pyproject | NumPy1.26.4 pin、Python>=3.11を確認。ただしQiskit等を含む広い研究環境 | 最小B pathに必要なNumPyだけを採用。editable installやF1用package一括導入はしない |

新規helperが必要だった理由は、B用の7field環境record、wheelとinstalled RECORDのbytes一致、private/public分離、mapped ELFのDT_NEEDED/SONAME照合を同時に行う既存の適合validatorを見つけられなかったため。hash実装を別に作り直さず、上記の逐字helperを使った。SOURCE provenanceは[REFERENCE_PROVENANCE.json](REFERENCE_PROVENANCE.json)。

既存testのPASSは今回の資格に流用していない。今回の科学SOURCE/test import・実行は0。環境の技術検査を新規に実施した。
