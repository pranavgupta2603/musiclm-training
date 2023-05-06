from audiolm_pytorch import AudioLM
from musiclm_pytorch import MuLaNEmbedQuantizer
from musiclm_pytorch import MusicLM

# setup the quantizer with the namespaced conditioning embeddings, unique per quantizer as well as namespace (per transformer)

quantizer = MuLaNEmbedQuantizer(
    mulan = mulan,                          # pass in trained mulan
    conditioning_dims = (1024, 1024, 1024), # say all three transformers have model dimensions of 1024
    namespaces = ('semantic', 'coarse', 'fine')
)

audiolm = AudioLM(
    wav2vec = wav2vec,
    codec = soundstream,
    semantic_transformer = semantic_transformer, #pass in the sematic transformer trained
    coarse_transformer = coarse_transformer, #pass in the coarse transformer trained
    fine_transformer = fine_transformer #pass in the fine transformer trained
)


musiclm = MusicLM(
    #combine audiolm and the mulan embed quantizer
    audio_lm = audio_lm, 
    mulan_embed_quantizer = mulan_embed_quantizer
)

music = musiclm('an electric guitar playing a soft rhythm near the beach with low background sounds of ocean waves.', num_samples = 1) #generate 1 audio sample for this text.
