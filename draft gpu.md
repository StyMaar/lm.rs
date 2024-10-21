Tensor = buffer_index, range_start, range_end

matmul => 
    - prend 3 Tensor: (v, out, Matrix)
    - vérifie que les dimensions sont correctes, et si ça match on fait la multiplication

load_buffer_to_gpu(buffer_index) (est-ce qu'on peut updater juste une range? Sans doute en vrai)

get_buffer_from_gpu(buffer_index, range_start, range_end)

Les matrices:
- ne sont pas mutables
- doivent être stockées à la fois sur la VRAM du GPU et dans le mmap-ed file accessible depuis le CPU 
- sont groupées au sein de gros tableaux (les weights)
=> elles doivent avoir leur propre abstraction (`Tensor` est une mauvaise abstraction)
=> à l'initialisation on copie les données dans un buffer de la VRAM et on garde une slice côté CPU.
=> lorsqu'on veut effectuer une opération côté CPU, on utilise la méthode `data()` pour avoir une slice immutable.
=> idéalement elles sont toutes stockées dans le même buffer avec juste un offset et une taille stocké dans la struct.
=> quand on veut effectuer une opération côté GPU, on passe (offset/size) en paramètre

Les vecteurs:
- sont mutables
- subissent plein de calculs sur le CPU
**- peuvent être sub-sliced** :
    - avant d'être utilisé comme out_vec de la matmul(kv_cache et value_cache)
    - pour être utilisé en lecture seule (q et key_cache)
- il faut faire des aller-retour entre le CPU et le GPU:
    - si le contenu du GPU est à jour, on execute le calcul
    - si la données a été modifiée sur le cpu, alors on ré-upload sur le GPU avant de faire le calcul
    - si on a besoin de la donnée sur CPU alors on la récupère sur le GPU
- pour `kv_cache` et `value_cache`on a besoin d'extraire des sous-vecteurs sur lesquelles effectuer des opérations GPU
    => idéalement il faudrait que cette opération prenne une référence exclusive au parent, mais je ne sais pas si ça peut marcher.
    => les sous-vecteurs doivent avoir une implémentation de drop qui met à jour le contenu du GPU (comme ça le tenseur parent reste cohérent)


partie py-rate Mael:
yx1bpHj

