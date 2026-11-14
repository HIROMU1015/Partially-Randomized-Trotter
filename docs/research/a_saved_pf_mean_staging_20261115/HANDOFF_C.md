# A → C：決定論PF shortlistの信号の受入準備

[REPORT](REPORT.md)／[STAGING_STATUS.csv](STAGING_STATUS.csv)（候補×round×X/Y）。

## 1. 受入準備の状況

| 候補 | configuration | S2（packet化済み） | S2L | S3 |
|---|---|---|---|---|
| Full PF S2・16h/7 | `11cf8ca6677a7ce651a00d827d0948e758510f95876acd24bf4d826c2bb6ca5d` | m0、m1、m4、m12 | m3 | m2、m5〜m11 |
| Full PF Y4・16h/3 | `a799697fc8ca…` | m0、m1、m2、m4、m12 | m3 | m5〜m11 |
| Full PF M5・16h | `70f7b31744ec…` | m0〜m4、m12 | — | m5〜m11 |
| B0I p7 S2・16h/7 | `a6e557770a2e4d96585ad5ca5903c4626ae47dda9775e9a9babc756149c38d84` | m0、m1、m4、m12 | m3 | m2、m5〜m11 |
| B0I p7 Y4・16h/3 | `defc5e88ff4d…` | m0、m1、m2、m4、m12 | m3 | m5〜m11 |
| B0I p7 M5・16h | `424eecbfdad0…` | m0〜m4、m12 | — | m5〜m11 |

- configurationの全ID・各roundのphysical ID・canonical IDは、STAGING.jsonにある。
- **価格shortlistとのphysical IDの対応**：CELL_PLAN_V2のm4〜m6の全cell（6候補×3 round×X/Y＝36 cell）で一致した（test 2）。
- 信号側の受入準備が済んでいるのはm4とm12だけである。m5・m6は、信号側ではS3（新しいmeanが必要）である。

## 2. S2Lを再利用できるか

- Track Aのm3（4件）は、入力・physical IDが一致する。
- ただし現行のpower・norm gate、backendの同一性、bba15083 reference pairに対する証拠がない。既存記録だけでは受入できないので、`PENDING_NOT_PROMOTED`とした。
- 解決策は、新しいpower・gateの計算（科学計算・承認待ち）か、ユーザーの判断である。

## 3. 最小の追加信号取得

- 44 round meanが必要（S3）。条件付きで、S2Lの扱いにより＋4、m0の由来の判断により＋2。
- 価格側のm4〜m6 cell（較正とholdout）と同じ候補が、全13roundで揃うのは、この計算の後である。

## 4. Cへのお願い（作業の重複を避けるため）

- 候補対応表を作るときは、この一覧と[A coverage](../a_signal_candidate_identity_coverage_20261114/SIGNAL_COVERAGE.csv)を参照し、候補の一覧を作り直さないでほしい。
- Truncated S2・0.1（信号は揃っているが価格がない）と、shortlist（価格はあるが信号が未完成）を、同じGに結合しないでほしい。
