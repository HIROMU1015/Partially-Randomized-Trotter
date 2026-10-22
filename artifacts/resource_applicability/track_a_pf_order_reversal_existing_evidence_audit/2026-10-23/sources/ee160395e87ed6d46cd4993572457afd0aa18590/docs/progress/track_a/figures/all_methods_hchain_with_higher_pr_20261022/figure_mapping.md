# 図と保存recordの対応

全費用図はfigure_data.jsonのrecords / figures / display_pointsを使用する。
figure_data.csvはdisplay_pointsの同一行をflat化したもの。元sourceは各record.sourceの固定commit/path/SHA/locator。
ratioはdisplay_exact、元totalはcost_exact。nullは理由付きの未表示枠で、0ではない。

| 図base（各PNG/SVG/PDF） | scope | mode | ε | CSV描画行（1起点） | 分母 |
|---|---|---|---:|---|---|
| `A_usual_signed_absolute_eps_0p05` | A | USUAL_SIGNED | 0.05 | 1–36 | absolute total |
| `A_usual_signed_relative_eps_0p05` | A | USUAL_SIGNED | 0.05 | 37–72 | LEGACY_FULL / PR_S2 |
| `A_usual_signed_absolute_eps_0p01` | A | USUAL_SIGNED | 0.01 | 73–108 | absolute total |
| `A_usual_signed_relative_eps_0p01` | A | USUAL_SIGNED | 0.01 | 109–144 | LEGACY_FULL / PR_S2 |
| `A_usual_signed_absolute_eps_0p005` | A | USUAL_SIGNED | 0.005 | 145–180 | absolute total |
| `A_usual_signed_relative_eps_0p005` | A | USUAL_SIGNED | 0.005 | 181–216 | LEGACY_FULL / PR_S2 |
| `A_usual_signed_absolute_eps_0p001` | A | USUAL_SIGNED | 0.001 | 217–252 | absolute total |
| `A_usual_signed_relative_eps_0p001` | A | USUAL_SIGNED | 0.001 | 253–288 | LEGACY_FULL / PR_S2 |
| `A_usual_signed_absolute_eps_0p0005` | A | USUAL_SIGNED | 0.0005 | 289–324 | absolute total |
| `A_usual_signed_relative_eps_0p0005` | A | USUAL_SIGNED | 0.0005 | 325–360 | LEGACY_FULL / PR_S2 |
| `B_usual_signed_absolute_eps_0p05` | B | USUAL_SIGNED | 0.05 | 361–369 | absolute total |
| `C_usual_signed_absolute_eps_0p05` | C | USUAL_SIGNED | 0.05 | 370–379 | absolute total |
| `B_usual_signed_relative_eps_0p05` | B | USUAL_SIGNED | 0.05 | 380–388 | PR_S2 |
| `C_usual_signed_relative_eps_0p05` | C | USUAL_SIGNED | 0.05 | 389–398 | M5_SUBSET |
| `B_usual_signed_absolute_eps_0p01` | B | USUAL_SIGNED | 0.01 | 399–407 | absolute total |
| `C_usual_signed_absolute_eps_0p01` | C | USUAL_SIGNED | 0.01 | 408–417 | absolute total |
| `B_usual_signed_relative_eps_0p01` | B | USUAL_SIGNED | 0.01 | 418–426 | PR_S2 |
| `C_usual_signed_relative_eps_0p01` | C | USUAL_SIGNED | 0.01 | 427–436 | M5_SUBSET |
| `B_usual_signed_absolute_eps_0p005` | B | USUAL_SIGNED | 0.005 | 437–445 | absolute total |
| `C_usual_signed_absolute_eps_0p005` | C | USUAL_SIGNED | 0.005 | 446–455 | absolute total |
| `B_usual_signed_relative_eps_0p005` | B | USUAL_SIGNED | 0.005 | 456–464 | PR_S2 |
| `C_usual_signed_relative_eps_0p005` | C | USUAL_SIGNED | 0.005 | 465–474 | M5_SUBSET |
| `B_usual_signed_absolute_eps_0p001` | B | USUAL_SIGNED | 0.001 | 475–483 | absolute total |
| `C_usual_signed_absolute_eps_0p001` | C | USUAL_SIGNED | 0.001 | 484–493 | absolute total |
| `B_usual_signed_relative_eps_0p001` | B | USUAL_SIGNED | 0.001 | 494–502 | PR_S2 |
| `C_usual_signed_relative_eps_0p001` | C | USUAL_SIGNED | 0.001 | 503–512 | M5_SUBSET |
| `B_usual_signed_absolute_eps_0p0005` | B | USUAL_SIGNED | 0.0005 | 513–521 | absolute total |
| `C_usual_signed_absolute_eps_0p0005` | C | USUAL_SIGNED | 0.0005 | 522–531 | absolute total |
| `B_usual_signed_relative_eps_0p0005` | B | USUAL_SIGNED | 0.0005 | 532–540 | PR_S2 |
| `C_usual_signed_relative_eps_0p0005` | C | USUAL_SIGNED | 0.0005 | 541–550 | M5_SUBSET |
| `B_noncancellation_absolute_eps_0p05` | B | NONCANCELLATION | 0.05 | 551–559 | absolute total |
| `C_noncancellation_absolute_eps_0p05` | C | NONCANCELLATION | 0.05 | 560–569 | absolute total |
| `B_noncancellation_relative_eps_0p05` | B | NONCANCELLATION | 0.05 | 570–578 | PR_S2 |
| `C_noncancellation_relative_eps_0p05` | C | NONCANCELLATION | 0.05 | 579–588 | FULL_M5_ONLY |
| `B_noncancellation_absolute_eps_0p01` | B | NONCANCELLATION | 0.01 | 589–597 | absolute total |
| `C_noncancellation_absolute_eps_0p01` | C | NONCANCELLATION | 0.01 | 598–607 | absolute total |
| `B_noncancellation_relative_eps_0p01` | B | NONCANCELLATION | 0.01 | 608–616 | PR_S2 |
| `C_noncancellation_relative_eps_0p01` | C | NONCANCELLATION | 0.01 | 617–626 | FULL_M5_ONLY |
| `B_noncancellation_absolute_eps_0p005` | B | NONCANCELLATION | 0.005 | 627–635 | absolute total |
| `C_noncancellation_absolute_eps_0p005` | C | NONCANCELLATION | 0.005 | 636–645 | absolute total |
| `B_noncancellation_relative_eps_0p005` | B | NONCANCELLATION | 0.005 | 646–654 | PR_S2 |
| `C_noncancellation_relative_eps_0p005` | C | NONCANCELLATION | 0.005 | 655–664 | FULL_M5_ONLY |
| `B_noncancellation_absolute_eps_0p001` | B | NONCANCELLATION | 0.001 | 665–673 | absolute total |
| `C_noncancellation_absolute_eps_0p001` | C | NONCANCELLATION | 0.001 | 674–683 | absolute total |
| `B_noncancellation_relative_eps_0p001` | B | NONCANCELLATION | 0.001 | 684–692 | PR_S2 |
| `C_noncancellation_relative_eps_0p001` | C | NONCANCELLATION | 0.001 | 693–702 | FULL_M5_ONLY |
| `B_noncancellation_absolute_eps_0p0005` | B | NONCANCELLATION | 0.0005 | 703–711 | absolute total |
| `C_noncancellation_absolute_eps_0p0005` | C | NONCANCELLATION | 0.0005 | 712–721 | absolute total |
| `B_noncancellation_relative_eps_0p0005` | B | NONCANCELLATION | 0.0005 | 722–730 | PR_S2 |
| `C_noncancellation_relative_eps_0p0005` | C | NONCANCELLATION | 0.0005 | 731–740 | FULL_M5_ONLY |

各panel内の行順はseries順、その内部はH-chain順。A左/右はdomainを分ける。
Dの2図はcoverage_audit.json（Aの160枠）および固定readiness情報を使い、費用CSVの行を持たない。
sourceのcommitは保存blobの所在、scientific_SOURCEは既存実装の識別子で別の役割を持つ。
