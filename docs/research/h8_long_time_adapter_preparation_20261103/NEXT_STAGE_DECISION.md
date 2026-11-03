# 次段階判断

2026-11-03。**PREPARATION_COMPLETE / SCIENTIFIC_EXECUTION_HOLD / mandatory STOP**。

1. **Adapter実装・人工検証**：Stage A/B/Cのlazy実行経路を実装し、人工・mock 21件PASS。実H8経路のend-to-end検証は今後の科学scope内で行う。
2. **保存入力**：SERVER2に固定H8 snapshot、DF/state receipt、保存Full28 preparation、prefix7/8 identity、DBA p15のbindingとp16のexact scalar receiptがある。選択入力のhashを照合した。内部path・seed・rawは非公開。
3. **新時間referenceの前提**：入力・sparse実装・環境・新保存領域・実行interfaceは準備済み。T1.6 referenceは未生成で、別の科学実行許可が必要。旧T.8結果による代用は拒否する。
4. **第一条件**：linear H8、1Å、STO-3G、保存tol-only1e-8 DF28、T1.6、ε.0005、P0。PR prefix7/8のS2対m5、Full28とB0/B0I prefix15/16のS2/Y4/m5を対照とする。候補domainは[科学実行計画](SCIENTIFIC_EXECUTION_PLAN.md)。第一batchはPR12＋PF45＝57候補の提案一覧をconfigurationへ保存した。探索n=9は提案。worker数とStage B選択規則を実行前に具体化する。
5. **必要な許可**：公開した新adapter commitをSOURCEとし、[固定入力](VALIDATION.json)、候補scope、Stage Aのrestore/scalar/reference/primitive/signal/shotと、選択後Stage Bの新sampling/build/control/compile/paired Gまでの実行許可。通常debugは同scopeでまとめて扱い、承認証明packetを作らない。
6. **追加の科学判断**：探索後にweighted paired Gの分散と希少orderの限界を踏まえ、確認精度目標・n・候補・比較方向を事前固定する。Stage Cは今はn未定でHOLD。同q診断や条件拡張、既存STOP変更は自動実行しない。p15 B0Iのexact scalar・p16 receipt一致と新time作用はStage Aで確認する。
7. **次の担当**：ユーザーがStage A/Bの具体的科学scopeと探索nを判断して明示許可する。Codexが許可後に固定scopeを作成しA→必要Bを実行、確認n案を示す。今回その計算は開始しない。

旧p8 flagのfalseは保存キー不在による比較未実施として説明済みで、変更していない。
旧rare-RTE正式終了、独立確認、H46研究のSTOP・grant・ledger・marker・retry条件を維持する。
新研究は別scope。科学性能・PR優位・最終総costの新結論はない。
