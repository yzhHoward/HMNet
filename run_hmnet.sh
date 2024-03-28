for model in HMNet
do

# ETTm2
for preLen in 96 192 336 720
do
labelLen=`expr $preLen / 2`
python -u run.py \
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTm2.csv \
  --task_id ETTm2 \
  --model $model \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $preLen \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --itr 3
done

# electricity
for preLen in 96 192 336 720
do
labelLen=`expr $preLen / 2`
python -u run.py \
  --is_training 1 \
  --root_path ./data/electricity/ \
  --data_path electricity.csv \
  --task_id ECL \
  --model $model \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $preLen \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 3 \
done

# exchange
for preLen in 96 192 336 720
do
labelLen=`expr $preLen / 2`
python -u run.py \
 --is_training 1 \
 --root_path ./data/exchange_rate/ \
 --data_path exchange_rate.csv \
 --task_id Exchange \
 --model $model \
 --data custom \
 --features M \
 --seq_len 96 \
 --label_len 48 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 8 \
 --dec_in 8 \
 --c_out 8 \
 --des 'Exp' \
 --itr 3
done

# traffic
for preLen in 96 192 336 720
do
labelLen=`expr $preLen / 2`
python -u run.py \
 --is_training 1 \
 --root_path ./data/traffic/ \
 --data_path traffic.csv \
 --task_id traffic \
 --model $model \
 --data custom \
 --features M \
 --seq_len 96 \
 --label_len 48 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 862 \
 --dec_in 862 \
 --c_out 862 \
 --des 'Exp' \
 --itr 3
done

# weather
for preLen in 96 192 336 720
do
labelLen=`expr $preLen / 2`
python -u run.py \
 --is_training 1 \
 --root_path ./data/weather/ \
 --data_path weather.csv \
 --task_id weather \
 --model $model \
 --data custom \
 --features M \
 --seq_len 96 \
 --label_len 48 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 21 \
 --dec_in 21 \
 --c_out 21 \
 --des 'Exp' \
 --itr 3
done

done
