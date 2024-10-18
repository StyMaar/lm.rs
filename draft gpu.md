Tensor = buffer_index, range_start, range_end

matmul => 
    - prend 3 Tensor: (v, out, Matrix)
    - vérifie que les dimensions sont correctes, et si ça match on fait la multiplication

load_buffer_to_gpu(buffer_index) (est-ce qu'on peut updater juste une range? Sans doute en vrai)

get_buffer_from_gpu(buffer_index, range_start, range_end)

Les matrices:
- ne sont pas mutables
- sont groupées au sein de gros tableaux (les weights)
=> elles doivent avoir leur propre abstraction (`Tensor` est une mauvaise abstraction)

Les vecteurs:
- sont mutables
- subissent plein de calculs sur le CPU
**- peuvent être sub-sliced** :
    - avant d'être utilisé comme out_vec de la matmul(kv_cache et value_cache)
    - pour être utilisé en lecture seule (q et key_cache)
