# A → B：RMSE科学実行前preflight

基点：A `510fbacd01a2d855202de4c045d8ba1972e5ff3d`、B final `82bc774a92b65cac06b88ebcb2c6979c2726253b`。旧A `b_handoff`（d562草案用）は変更していない。本追加は実mean/N/RMSEを生成しない。

## API

```python
from trottertracks import pr_rpe_rmse_preflight as a
inventory = a.saved_inventory(repo_root)       # pinned Git field-presence only
result = a.dry_run(B_configuration, A_schedule, inventory, root=repo_root)
```

CLI（fresh outputのみ）:

```bash
python3 -B scripts/preflight_pr_rpe_rmse.py --inventory-only --out /tmp/a_rmse_inventory.json
python3 -B scripts/preflight_pr_rpe_rmse.py --configuration fixed_B.json --schedule fixed_A.json --inventory /tmp/a_rmse_inventory.json --out /tmp/a_rmse_dry_run.json
```

`fixed_B.json`はB generateのconfiguration全体、`fixed_A.json`はA schedule全体。M/κの新規選択・実tail normによる新schedule生成・科学評価はこのCLIでは行わない。固定SOURCEのGit objectsがローカルに必要。B生成/検証関数は82bc固定ASTをそのまま使用し、Bの作業branchは読まない。Bが新仕様を公開した場合、このpinの差分だけ別途確認する。

## 入力と出力

- inventory.mean descriptor：candidate/binding/structure/profile/source/saved_status、signal field名、allowance field名。値・旧Nを含めない。reference descriptor：binding/T/source/field名。8 carrierと基準taskの出所を記録。
- 全round候補、A request/configuration ID、B configuration/candidate ID、物理構成/時間/κ/λ/law SOURCEの照合。A/BのIDを同一視しない。
- `TECHNICAL_PREFLIGHT_COMPLETE_NOT_SCIENCE`はmetadata計画を作れたという意味。`BLOCKED`はSOURCE/構成/保存失敗の問題。どちらも科学的PASSではない。
- 各roundに保存再利用候補、欠測field、SOURCE拒否理由、承認後だけの追加call上限案。pairはshapeの存在を確認しただけで、数値gateは再確認していない。
- `N_packet`はB finalのidentity/law_bindingをそのままcopy。status=NOT_EVALUATED、evidence_kind=PREFLIGHT_ONLY、全N_X/YとRMSE boundはnull、全round資格=false。nominal Nも今回はnull。B assessが科学Nとして拒否することが正しい。
- packetの`rmse_evidence`はB finalで必要なsource/status/error_contract/rmse_bound_ha/covers_state_target_alias_prior_decoder_all_rounds/finite_K_status/signed_piece_kappa_statusへ欠測を対応付ける。科学証拠の代用にはしない。

## B側へ引き渡す接続差分

1. H4旧PFはB0/p7。B final full-H PFとは異なり、relabel不可。新PF条件の採択はユーザー判断と必要な科学scopeが要る。
2. H6 A scheduleはm0..3 p5/R32/q1/δ=Tを保持する。B finalの固定L_D=p7では表せず、external event mapperだけでも解決しない。Bで明示的なearly recipe表現を検討し、元SOURCEを保持する必要がある。AはB controllerを改修しない。
3. κはA長名とB PAPER_TOTAL_ROTATIONSのpaper E21定義を対応付ける。κ_Bと混同しない。λ値だけでなくSOURCE、lawの同一SOURCEまたは同一Git blobも照合する。
4. B rational timeとA binary64のexact値、ceil grid対固定δ、prefix/normalization/finite-K条件が一致する場合だけmean再利用候補になる。時間丸めを暗黙に修正しない。
5. A全bindingをBへ保全する。Bが追加するdecomposition/SOURCEとA証拠のjoin、prior/decoderとB error_contractの科学的対応は外部証拠として残る。
6. sampling law/implementation、厳密ground/state model、finite-K→RMSE、signed envelope、誤差合成、全round decoder統計が未資格。旧conditional PASS/α=.05は代用不可。

## 次の科学実行の範囲案（未承認）

1系・固定(M,κ)とRMSE policyを選び、2roundの局所pilotを提案する。H4 m0/m6またはH6 m4/m8はP1信号とreferenceが揃う参考点だが、共通κから両方のcandidateがP1と一致するとは仮定しない。保存値を使う数値gate/新policy再解析も明示承認が必要。

一致すれば保存再解析だけ、不一致/回収不能なら不足meanと内部対照のみ最大各2。pair不足時だけ2経路referenceを追加。phase/radius/biasと経験的nominal Nまでで、科学的適格N・全RMSE保証・C/G・κ最適化・samplingへ自動進行しない。全schedule評価はearly/full-H PF問題と科学的誤差契約を解決してから別scopeとする。

14新規＋17 RMSE＋12旧高確率型の43人工/回帰PASS。**Aは公開照合後mandatory STOP。ユーザーが次の有限科学scopeを承認するまで新科学callなし。**
