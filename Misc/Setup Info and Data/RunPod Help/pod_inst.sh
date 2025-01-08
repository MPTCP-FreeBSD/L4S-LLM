ssh root@69.30.85.20 -p 22135 -i ~/.ssh/id_ed25519


ssh root@69.30.85.20 -p 22057 -i ~/.ssh/id_ed25519

chmod 600 ~/.ssh/id_ed25519;chmod 600 ~/.ssh/id_ed25519.pub


sudo scp -P 3322 -p -i ~/.ssh/mptcprootkey -r NetLLM-Personal root@192.168.56.1:
sudo scp -334423 -p -i ~/.ssh/mptcprootkey -r NetLtLLM-Personal root@192.168.56.1:


scp -P 22008 -i ~/.ssh/id_ed25519 -r downloaded_plms root@69.30.85.42:/workspace/




sudo scp -P 22116 -p -i ~/.ssh/id_ed25519 -r NetLLM-Personal root@69.30.85.116:/workspace/



scp -P 22001 -p -i ~/.ssh/id_ed25519 -r root@69.30.85.111:/workspace/NetLLM/llm_framework /runpodmodelfiles
scp -P 22001 -p -i ~/.ssh/id_ed25519 -r root@69.30.85.111:/workspace/NetLLM/llm_framework/data /runpodresults
ssh root@69.30.85.111 -p 22001 -i ~/.ssh/id_ed25519


python run_plm.py --adapt --grad-accum-steps 32 --plm-type llama --plm-size base --rank 128 --device cuda:0 --device-out cuda:1 --lr 0.0001 --warmup-steps 2000 --num-epochs 80 --eval-per-epoch 2 
python run_plm.py --adapt --grad-accum-steps 32 --plm-type llama --plm-size base --rank 128 --device cuda:0 --device-out cuda:1 --lr 0.0001 --warmup-steps 200 --num-epochs 80 --eval-per-epoch 2



Test performance of NetLLM:
python run_plm.py --test --grad-accum-steps 32 --plm-type llama --plm-size base --rank 128 --device cuda:0 --device-out cuda:1 --lr 0.0001 --warmup-steps 2000 --num-epochs 80 --eval-per-epoch 2







ssh root@69.30.85.70 -p 22071 -i ~/.ssh/id_ed25519



scp -P 22071 -p -i ~/.ssh/id_ed25519 -r root@69.30.85.70:/workspace/L4S-LLM/llm_framework/Log Data /log_data

/workspace/L4S-LLM/llm_framework/Log Data