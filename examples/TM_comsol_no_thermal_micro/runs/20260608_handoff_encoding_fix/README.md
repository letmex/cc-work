# Handoff Encoding Fix

This package documents a workflow fix for the final ChatGPT sync command.

Problem: a previous final response emitted mojibake text such as `璇诲彇`,
`锛`, and `銆` instead of normal Chinese.

Fix: add explicit UTF-8 encoding guidance to the TM handoff workflow files.

No simulation, model, or result files were changed.

