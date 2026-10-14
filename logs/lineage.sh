while ! grep -q AFTER13B_DONE logs/register_13b.log 2>/dev/null; do sleep 30; done
zsh logs/register_run.sh "olmo@x7binst"
echo LINEAGE_DONE
