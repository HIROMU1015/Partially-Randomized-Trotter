# A → C/B：Full PF公式・actual幅・全roundの接続

[REPORT](REPORT.md)／[PLAN](PLAN.json)／[全195行CSV](ROUND_SCHEDULE.csv)／[未実行科学範囲](SCIENCE_PLAN.md)。科学実行0、公式/幅の最適候補・精度PASS・新C/N/Gは未取得。

## 候補metadata

S2/Y4/M5×要求Δ=.05/.1/.2/.4/.8の15配置。H4 DF10、全one-body＋DF10/prefix10、同H/state/snapshot/sector(2,2)/target・h.1/M12/P0。B437のcanonical IDをD e7d/C86の既存APIで生成し、S2/.1は元configuration f2d58f...と一致した。独自のcanonical ID又は物理cid規則を追加していない。

各`configurations[i]`のconstructorに元base manifest c546、D.full_pf_spec/B.generateの引数とSOURCE、configuration IDを保持する。そこからCは同じ完全V2 configurationを再現できる。各roundにはT exact/float、q_RPE/qPF、actual幅、canonical candidate ID、完全physical candidate、price source selectorを保存した。global共通binding/profile/targetは`shared`を使う。

`step_cells`はformula×actual幅15種類。195行・177 physical候補への対応は`uses/step_cell`に保持する。要求幅が違ってもearly roundで同physicalになる場合は、SOURCE/law/profile/scalar/axisを含む同値性を確認して価格の重複取得を避ける。configuration IDだけを付け替えて旧値を受入済みにしない。

## Cのcompiled価格証拠

C3aeの旧6 recipe coverageを最適menuとして流用せず、D e7dの確定catalog全136候補に今回15配置をmetadata照合した。390軸参照のうちEXACT38、catalog内MISSING18、照合域外334。値・係数を新集計したものではない。

`price_coverage.source`は正本catalogの具体的row locatorであり、原receipt/native SOURCE/axis fingerprintはそこから追跡する。EXACT表示は既存D判定の参照で、新consumer受入を意味しない。Cのprice prediction/mean/CI/qualificationは未取得を保持し、同physical IDだけで新ZZ lowering・新profileへ移植しない。旧PR C1係数やE[S]、q比例をFull PF欠測へ当てない。

S2/Y4/M5のpiece数1/3/11と`qPF×piece数`は**構造上の作用数**でcompiled RZではない。Cは同一candidate、全controlled measured X/Y wrapper、P0、元scalar/native/compiler/seedのdomainで接続する。今回Cのquote/model/fit/compileは呼んでいない。

## H_eff結果の将来受渡し

S2/.1はA69eeを参照する。新14stepは承認後に既存direct-expmとcache対応評価器で実行する。元Hの保存eigenvalues/vectorsをSOURCE付き`reference_eigensystem`へ供給し、同元H行列とresidual gateを必須とする。新eigh0、旧mean再解析0。旧state weightsはreuseせず同stateから計算し、失敗時に新eighへfallbackしない。

承認後のcache入力は既存出力のキーをそのまま使う。次の例は今回実行していない。

```python
cache = dict(
    source=plan['shared']['cached_reference_eigensystem'],
    hamiltonian=plan['shared']['binding']['hamiltonian'],
    eigenvalues_ha=saved_diagnostic['reference_comparison']['eigenvalues_ha'],
    eigenvectors=saved_diagnostic['reference_comparison']['eigenvectors'],
)
# evaluate(..., reference_hamiltonian=saved_context.full,
#          reference_eigensystem=cache)
```

将来のpacketはformula/actual幅・linked canonical/physical ID・branch・全spectrum/overlap・δE診断・state/law/numerical資格・SOURCEを維持する。原Hの最低energyとeffective minimumを自動同定せず、global RMSE/十分N資格と分離する。

**A**はPF spectral shiftと既定εtrotの局所screen・全round例外を提供する。**C**はその候補と同domainの価格coverage/価格取得計画を扱う。**B**がraw直接riskを使うなら同じPF δEを二重加算しない。BのN/MC/正式GをAで再実装せず、この物理探索をBの結果待ちにしない。

## 実行承認

**ユーザー**が固定15配置（新14 H_eff、直接/spectral各最大28、新元H eigh/reference/旧mean再解析0）の有限範囲を承認した後に**Codex A**が実行する。現段階では技術成果commit/push・remote照合後mandatory STOP。C/Bへ外部メッセージ送信はせず、本HANDOFFを公開する。
