# SOURCE・runtime・Admission v3

`source_inventory.json` に固定公開commit、測定SOURCE、21 measured filesとSHA256、保護SOURCEを記録する。
GitHub Appのcommit metadataにより公開commit SHAが測定SOURCEと違う場合も、runtime bytesは同じである。
`measurement_source_history.bundle` は基点1cd2a719をprerequisiteに、元の測定commitと途中履歴を復元する。

`b3_parallel_admission_v3` は旧 `b3_formal_admission_v2` と別のschema。
科学release/adoptionはnull、content_adopted/run_readyはfalse、contract_grants_executionはfalseで固定。
`r1_reference` は人工fixtureで作る旧R1構造の参照であり、その人工adopted表記を正式採択と扱わない。
新candidateは旧candidateにruntime SOURCE pinを足す以外の科学field差を拒否する。
設定、codec、observer、transport、逐次/並列条件、全input、schemaをformal fake bindingへ含める。

Runtime manifestは新6 roleと旧依存SOURCE pin、entrypoint path/API/hash、import位置、schema pinを束縛する。
interpreter実体SHA256、version/cache tag、json/fractions/multiprocessing/sqlite3等の実file SHA256も記録・再照合する。
親はadmission前後と各phase後、workerはconstructor前にSOURCE/runtimeを検査する。
人工rootには固定fileをコピーし、実import位置はruntime_rootへ固定する。両方のbytesを照合する。
人工factoryとbit関数の実file位置・hashもfake input/runtime manifestへ束縛する。

schema一式はoutput_schema/の7 JSON Schema＋protocol.json。全8 fileをinput pinとruntime/schema pinへ束縛した。
JSON Schemaは外形を定義し、予約順・hash・prefix・会計等の意味条件はprotocol.jsonおよびSOURCEの検査に従う。
旧R1 contract、旧host/admission/readiness、scientific sourceは上書きしない。

実approval/token/grant/claim、正式science release commit、実行可能な正式contract digestは作らない。
人工contract/receipt内のdigestはfake bundleの整合性検査値であり、実科学の承認credentialではない。
