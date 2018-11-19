export PS4='+{$LINENO($(date +"%F %T")):${FUNCNAME[0]}} '
full_path=$(readlink -f $0)
cd $(dirname $full_path)
exec 1>${full_path%.*}.log
exec 2>${full_path%.*}.err
set -x

lmplz -o 5 --skip_symbols "<unk>" < ../data/ptb.train.txt.UNK > ../model/5gram.arpa.unk
build_binary ../model/5gram.arpa.unk ../model/5gram.binary.unk

lmplz -o 4 --skip_symbols "<unk>" < ../data/ptb.train.txt.UNK > ../model/4gram.arpa.unk
build_binary ../model/4gram.arpa.unk ../model/4gram.binary.unk

lmplz -o 3 --skip_symbols "<unk>" < ../data/ptb.train.txt.UNK > ../model/3gram.arpa.unk
build_binary ../model/3gram.arpa.unk ../model/3gram.binary.unk

lmplz -o 2 --skip_symbols "<unk>" < ../data/ptb.train.txt.UNK > ../model/2gram.arpa.unk
build_binary ../model/2gram.arpa.unk ../model/2gram.binary.unk

set +x
