from typing import Tuple
import sys
# sys.path.append('/home/nlpfall2020/CSCI544_Project')

from . import registry

register = registry.register


@register
class UgaHebSmallNoSpe:
    lost_lang: str = 'uga-no_spe' #uga-no_spe
    known_lang: str = 'heb-no_spe' #heb-no_spe
    cog_path: str = 'data/uga-heb.small.no_spe.cog'
    num_cognates: int = 221
    num_epochs_per_M_step: int = 150
    eval_interval: int = 10
    check_interval: int = 10
    num_rounds: int = 10
    batch_size: int = 500
    n_similar: int = 5
    capacity: int = 3
    dropout: float = 0.3
    warm_up_steps: int = 5


@register
class linearB:
    lost_lang: str = 'linear_b' #uga-no_spe
    known_lang: str = 'greek' #heb-no_spe
    cog_path: str = 'data/linear_b-greek.names.cog'
    num_cognates: int = 455
    num_epochs_per_M_step: int = 150
    eval_interval: int = 10
    check_interval: int = 10
    num_rounds: int = 10
    batch_size: int = 500
    n_similar: int = 5
    capacity: int = 3
    dropout: float = 0.3
    warm_up_steps: int = 5


@register
class linearA:
    lost_lang: str = 'lineara' #uga-no_spe
    known_lang: str = 'greek' #heb-no_spe
    cog_path: str = 'data/linear_AB.cog'
    num_cognates: int = 455
    num_epochs_per_M_step: int = 150
    eval_interval: int = 10
    check_interval: int = 10
    num_rounds: int = 10
    batch_size: int = 500
    n_similar: int = 5
    capacity: int = 3
    dropout: float = 0.3
    warm_up_steps: int = 5


@register
class Japanese:
    lost_lang: str = 'zh' #uga-no_spe
    known_lang: str = 'ja' #heb-no_spe
    cog_path: str = 'data/zh_ja.cog'
    num_cognates: int = 919
    num_epochs_per_M_step: int = 150
    eval_interval: int = 10
    check_interval: int = 10
    num_rounds: int = 15
    batch_size: int = 500
    n_similar: int = 5
    capacity: int = 3
    dropout: float = 0.3
    warm_up_steps: int = 5


@register
class Chinese:
    lost_lang: str = 'ja' #uga-no_spe
    known_lang: str = 'zh' #heb-no_spe
    cog_path: str = 'data/zh_ja.cog'
    num_cognates: int = 919
    num_epochs_per_M_step: int = 150
    eval_interval: int = 10
    check_interval: int = 10
    num_rounds: int = 15
    batch_size: int = 500
    n_similar: int = 5
    capacity: int = 3
    dropout: float = 0.3
    warm_up_steps: int = 5
