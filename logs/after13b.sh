while ! grep -q OLMO13B_DONE logs/olmo13b.log; do sleep 30; done
zsh logs/register_run.sh olmo13b "olmo13b@inst"
echo AFTER13B_DONE
