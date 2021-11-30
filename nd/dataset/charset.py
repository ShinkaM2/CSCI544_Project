#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy

import torch
import numpy as np

from dev_misc import Map

PAD_ID = 0
SOW_ID = 1
EOW_ID = 2
UNK_ID = 3

PAD = '<PAD>'
SOW = '<SOW>'
EOW = '<EOW>'
UNK = '<UNK>'
START_CHAR = [PAD, SOW, EOW, UNK]

_CHARSETS = dict()


def register_charset(lang):
    global _CHARSETS

    def decorated(cls):
        assert lang not in _CHARSETS
        _CHARSETS[lang] = cls
        return cls

    return decorated


def get_charset(lang):
    '''
    Make sure only one charset is ever created.
    '''
    global _CHARSETS
    cls_or_obj = _CHARSETS[lang]
    if isinstance(cls_or_obj, type):
        _CHARSETS[lang] = cls_or_obj()
    return _CHARSETS[lang]


def _recursive_map(func, lst):
    ret = list()
    for item in lst:
        if isinstance(item, (list, np.ndarray)):
            ret.append(_recursive_map(func, item))
        else:
            ret.append(func(item))
    return ret


class BaseCharset(object):

    _CHARS = u''
    _FEATURES = []

    def __init__(self):
        self._id2char = START_CHAR + list(self.__class__._CHARS)
        self._char2id = dict(zip(self._id2char, range(len(self._id2char))))
        self._feat_dict = {}
        for f in self.features:
            self._feat_dict['char'] = None
            self._feat_dict[f] = False

    def __len__(self):
        return len(self._id2char)

    def char2id(self, char):
        def map_func(c): return self._char2id.get(c, UNK_ID)
        if isinstance(char, str):
            return map_func(char)
        elif isinstance(char, (np.ndarray, list)):
            return np.asarray(_recursive_map(map_func, char))
            # return np.asarray([np.asarray(list(map(map_func, ch))) for ch in char])
        else:
            raise NotImplementedError

    def id2char(self, id_):
        def map_func(i): return self._id2char[i]
        if isinstance(id_, int):
            return map_func(id_)
        elif isinstance(id_, (np.ndarray, list)):
            return np.asarray(_recursive_map(map_func, id_))
            # id_.tolist()
            # if id_.ndim == 2:
            #     return np.asarray([np.asarray(list(map(map_func, i))) for i in id_])
            # elif id_.ndim == 3:
            #     return np.asarray([self.id2char(i) for i in id_])
        else:
            raise NotImplementedError

    def get_tokens(self, ids):
        if torch.is_tensor(ids):
            ids = ids.cpu().numpy()
        chars = self.id2char(ids)

        def get_2d_tokens(chars):
            tokens = list()
            for char_seq in chars:
                token = ''
                for c in char_seq:
                    if c == EOW:
                        break
                    elif c in START_CHAR:
                        c = '|'
                    token += c
                tokens.append(token)
            return np.asarray(tokens)

        if chars.ndim == 3:
            a, b, _ = chars.shape
            chars = chars.reshape(a * b, -1)
            tokens = get_2d_tokens(chars).reshape(a, b)
        else:
            tokens = get_2d_tokens(chars)
        return tokens

    def process(self, word):
        # How to process chars in word. This function is language-dependent.
        raise NotImplementedError

    @property
    def features(self):
        return self._FEATURES


@register_charset('en')
class EnCharSet(BaseCharset):

    _CHARS = u'abcdefghijklmnopqrstuvwxyz'
    _FEATURES = ['capitalization']

    def process(self, word):
        ret = [copy.deepcopy(Map(self._feat_dict)) for _ in range(len(word))]
        for (i, c) in enumerate(word):
            if c in self._char2id:
                ret[i].update({'char': c})
            else:
                c_lower = c.lower()
                if c_lower in self._char2id:
                    ret[i].update({'char': c_lower})
                    ret[i].update({'capitalization': True})
                else:
                    ret[i].update({'char': ''})
        return ret


@register_charset('es')
class EsCharSet(BaseCharset):

    _CHARS = u'abcdefghijklmnñopqrstuvwxyz'
    _FEATURES = ['capitalization']


@register_charset('es-ipa')
class EsIpaCharSet(BaseCharset):

    _CHARS = u'abdefgiklmnoprstuwxúɲɾʎʝʧ'
    _FEATURES = ['']


@register_charset('it')
class ItCharSet(BaseCharset):

    _CHARS = u'abcdefghijklmnopqrstuvwxyzàèéìïòöù'
    _FEATURES = ['capitalization']


@register_charset('it-ipa')
class ItIpaCharSet(BaseCharset):

    _CHARS = u'abdefghijklmnopqrstuvwzŋɔɛɲʃʎʤʧ'
    _FEATURES = ['']


@register_charset('pt')
class PtCharSet(BaseCharset):

    _CHARS = u'abcdefghijklmnopqrstuvwxyzáâãçéêíóôú'
    _FEATURES = ['capitalization']


@register_charset('pt-ipa')
class PtIpaCharSet(BaseCharset):

    _CHARS = u'abdefgiklmnopstuvzäɐɔɛɨɾʁʃʎʒ'
    _FEATURES = ['']


@register_charset('heb')
class HebCharSet(BaseCharset):

    _CHARS = u'#$&-<HSTabdghklmnpqrstwyz'
    _FEATURES = ['']


@register_charset('uga')
class UgaCharSet(BaseCharset):

    _CHARS = u'#$&*-<@HSTZabdghiklmnpqrstuvwxyz'
    _FEATURES = ['']


@register_charset('heb-no_spe')
class HebCharSetNoSpe(BaseCharset):

    _CHARS = u'$&<HSTabdghklmnpqrstwyz'
    _FEATURES = ['']


@register_charset('uga-no_spe')
class UgaCharSetNoSpe(BaseCharset):

    _CHARS = u'$&*<@HSTZabdghiklmnpqrstuvwxyz'
    _FEATURES = ['']


@register_charset('el')
class ElCharSet(BaseCharset):

    _CHARS = u'fhyαβγδεζηθικλμνξοπρςστυφχψω'
    _FEATURES = ['']


@register_charset('greek')
class ElCharSet(BaseCharset):

    _CHARS = u'fhyαβγδεζηθικλμνξοπρςστυφχψω'
    _FEATURES = ['']


@register_charset('linb-latin')
class LinbLatinCharSet(BaseCharset):

    _CHARS = u'23adeijkmnopqrstuwz'
    _FEATURES = ['']


@register_charset('lineara')
class LinACharset(BaseCharset):
    _CHARS = u'𐀃𐀳𐀊𐘞𐀨𐘇𐀙𐀚\U0001076b𐀓𐀕𐀲𐀇𐄉𐀴𐀉𐀏𐀂𐀫𐀶𐀠𐀖𐀞𐀅𐀬𐀷𐀛𐀱𐀑𐘭𐀼𐀮𐀔𐙫𐘦𐘙𐘟𐀯𐘺𐀤𐘉𐙠𐘽𐀐𐘶𐘛𐙕𐀁𐘜𐙹𐘯𐙀𐙱𐘐𐘹𐘄𐙖𐘣𐘰𐀢𐚗𐘷𐙙𐀘𐘩𐙋𐙯𐘧𐙄𐘊𐙛𐘨𐙇𐚢'
    _FEATURES = ['']


@register_charset('linear_b')
class MinoanCharSet(BaseCharset):

    _CHARS = u'𐀀𐀁𐀂𐀃𐀄𐀅𐀆𐀇𐀈𐀉𐀊𐀋𐀍𐀏𐀐𐀑𐀒𐀓𐀔𐀕𐀖𐀗𐀘𐀙𐀚𐀛𐀜𐀝𐀞𐀟𐀠𐀡𐀢𐀣𐀤𐀥𐀦𐀨𐀩𐀪𐀫𐀬𐀭𐀮𐀯𐀰𐀱𐀲𐀳𐀴𐀵𐀶𐀷𐀸𐀹𐀺𐀼𐀽𐀿𐁀𐁁𐁂𐁄𐁅𐁆𐁇𐁈𐁉𐁊𐁋'
    _FEATURES = ['']


@register_charset('fr')
class FrCharSet(BaseCharset):

    _CHARS = u'abcdefghijklmnopqrstuvwxyz'
    _FEATURES = ['capitalization']


@register_charset('lost')
class LostCharSet(BaseCharset):
    _CHARS = u'aeioubpdtfvgklmnrszw'


@register_charset('k1')
class K1CharSet(BaseCharset):
    _CHARS = u'aeioubpdtfvgklmnrszw'


@register_charset('k2')
class K2CharSet(BaseCharset):
    _CHARS = u'aeioubpdtfvgklmnrszw'


zh = u'晓卓觀马大经記叶樓身沓妖淑岳箕互幣隊动症勤語點塞玫朝电总拼因春浦空相楊聖商少毛产领腳乡東萧日紀音那水術原年扩县經为琉交舒苏單机壽式畴都乐状隆太物複集间廟造歷組郑萇恋央甲禁錄态册资干汽投燈旨径向围尔濮上任屋斗絹字輕奉奧虹們皇伯行坊附役時礎版開能薇荒灭廢特基院治型灵頂唱拖庄欧圖祖酿唇餘須纬改沭盾紙昭兵摩應？陸棋后計眾千狱数足體域老阴鮮馬合俠溫陣親鎮盛炎灣绿球刘雫急景彌挑均戴壁再归菁公智科黨亞、第首烟植泽团國独柳味庆羽專西堅香鶴縣獻亹縱铁符催康乃曲執越閣進昙報臺区里粟腊羅欢号論俁纯片伊本比吳祭漂吾風业切共台成巢雜壤权秦项理載航克示俸溪裡備步灾崎校直離前鐘汉控月權云掖曹展滦夢忍地口择园恒蛮宁洲汤划张護会爭代小广工指官宗书华坪品岡群侏兒吉滿對整阶部幕心衔色址诸冈釜命讓钵使标幹阵沒北加泉宿豐综已警表昌冰偵熊期古器祥八省度件达雙似勅帝京辯連關常丘唯韦案網启區女花宙授潘軍语药九贴东現何史导譴救與則世名孙影發參番定正播锦樂次誌好說媛居陈魏跑蓝鰍矢童折獎坛体橿骨庙雨隅近仁目紅和国沼衣隸六纪明员立功湖處僑元化覽接半议批术铺德银膜車漫詞藍他论善州函圍美节摸叫装象療果鹽內貫迪联构修尼军天兰升中沈柴堂英政遠敲我貓援放藝橋納参諸見郵恆貴堀塚員濟义况遗解秀遇摺通捷余延力管坡村圣眼技陳遣望引歌這間冠 值茂算硕靜陽分宣塵龜幟开守市倉奈松溝文盟阳澤礼七森剩號牧质韩場优告玉推杯家館排儒數探利園兴彥真機楸农督长吹蓟门濱志櫟火攻晋食新委子肃博四鄉海榮场團类选萬泾鬼師幡遊頹铜作令萩轨標電连慶外顾万情致亭恩重富鍾葵巴孩御尾舍的泰襟震洞鮫息题薩變纳察者丹初济艦城递江距坦織庭動稻在頭巫來受訴錢氣局韵憾戶处鯖周又也粉堡印盐鄰让报友木卫封佛寺葉街陆娄即寿藤狐醫磺戰滨野猿贵鷹量言畫狹竈草別端哪鬥红道图鴻笹寨席予时曆请石际汀统貨坂浮豆岬邊雷張货宇武收浅始持φ無韓總族同带鱼頷協纲獨务路蠶結赤林伸築熱鹏深风雪話甚蹤谷事彙雅迟殺蘭疏灘陀錦固府弘秋會自鐵士穗達鹿觉肥助赵驻屬头愛挺旋览室精贖皋赛券判促板托鈴南旅边媽界碳川尻念效来胀櫻瀨勞座埃私白策船五渤荷债夏埔发學塰除弃等糸方司穎支由橫全賽係劳勐所轉桥傳鲜龍双永塘彈组育厅胞艺神圓观門以螺是波求匯查劃陶杜賣凭战湯將選缺一舞仲益丁有建津根郭夜喜平爆晶超温之起律細主飛不协若营非将服麥轮社鳥篇线内就沙隶呼熟用佐桃淺宴打密医得笔洋疗堆牌軒车信替聞幸雄钦及飯问嘴實嚴惠驿岁流普項与聪兩午典淳山宫氧青凱苑賀賓黑纹高議畠手父系三久鸣芦抗際入最龙土星制虎衛柔港田民條興竹荔奖镇凯无证存線篤奴梅瑰知威人你王歐河途性形岩票縮角靖单维馆考瑞俄藩站菲歡多酒徵湘帶然争郎洗慈关弓渡坐母郡出硫燃结町鳳挂劇积防-米种宮薔短決谱雲統各購队沃邮義課快仙良病程布梨敷長清簗金话岱唤率潮调毒聯騎应法秘習榆舉運證二下党底旗列琴词气学福十既生教稀貿历拉積押勇现廣履计擊招贸速視安师源觸失遙假逢位架映潜可後丸保華井织面務島缩岭級串储登黃塔晝徽阪岛曉光反'


@register_charset('zh')
class ChineseSet(BaseCharset):
    _CHARS = zh


ja = u'卓茘遺大魔記要沓頃妖淑岳箕仏類幣隊症語曇え〜霊韻舎朝と灯少?春空浦恵聖相因楊）商毛順東紀日説寧音那冊水術原交年琉舒釈歴式畴都暁薊状隆俣客太ぎ物複集軟調廟造恋組萇央甲禁沢汽孫よ投旨径向任上濮屋斗牽絹字置奉虹皇伯行坊奇附時観ら役礎関版能画荒特徳院基露旧治型な頂聡豊欧祖唇沭改盾紙昭兵題陸棋計后千数足録棵域盤老馬合実軽鉄陣親鎮盛導炎球魚栄呪雫急児書景戴満壁再菁公智科応哲照、邸第首植独羽柳味西香鶴ぬ亹康猟層乃優執越閣曲進報区里粟号論図（亜片伊本横っ風像比暴挙吾漂切ぺほ共活台彼成氷気秦瀋理航克弁俸確巣備済戸営崎校直離前鐘専獄月鯨掖夢忍展れば退地口だ恒洲蚕発護会指代小官工宗坪慮岡写群侏吉脚触整部籍幕心廃色釜渋命緯使幹待み介北加泉宿せ偵警昌表試熊期古け器祥八省件度似勅戦京帝連常対丘網嶺宙女花授陰潘変軍炉贋九寝廓種条史…を経容救則世名影庁番定播魏次誌好媛居鰍矢童折体橿領仕啓顕骨近隅銭仁稲目衆紅和国沼六明鮮立湖す材歩元化僑顎半証ゅ査買施膜点批車漫藍他善詞顧団州び美さげ奨具装象亀療酵果厳君貫択棍修耽芸炭尼天中業柴堂岐英鋪政広放援賞橋納参諸見譜郵貴婁堀酸帰員塚殊秀解余通捷延覧管力坡村滅眼麗賊陳遣縦引技お望伝むど冠間 塩て茂算宣陽分塵ぼ正守文市奈松溝狭盟倉七礼財森拡牧巡場皐告構推に玉う館家利惑杯県園ぐ排儒真探機楸督囲垢労吹閥志火攻晋食新委子博問四海が辺幡遊師ち両漢作令萩標鉢売電慶舌外姫呉配万情致亭恩重富わ御尾的泰襟震洞鮫息薩円察者初艦城百江距嘘織庭動頭在巫そ継昼受訴局る暦鯖周又緑堡監逆木封佛寺葉街為つ即藤寿狐あ猿野荘鷹量言竈草はも別端権道鴻笹寨予ろ維石貨汀坂総豆岬張宇武浅止始持φ無滑こ韓焦か同瀬駅設撃協弥路結赤林腸歓築熱深趙話雪甚谷事彙終産掛筆素殺蘭灘陀錦固府弘秋徴自彦士鹿責肥愛挺旋促室精券判蘇板鈴南働界た収川尻念来櫻座私白策船五夏埔媒農塰除等糸司方穎支供由銀全係所勐龍双仰続永塘育胞神門以碁災単波準杜湯剰選銅一込舞奥益丁有建津根郭夜楡平い晶矛温へく匂之綱律細ひ主飛不若将非付服で譲社鳥転内沙ま質呼縄用佐桃竜打密医洋軒替信聞鳴幸雄欽ふ飯隷嘴づ節ん浜流普矩項与障典淳山帯青凱苑賀高議軌畠臘手父系三久入芦抗際債最土星壌虎衛制残柔涇港田民興竹脱憶声般線篤闘奴梅知り威薬人王河候性形穂遅岩縮し射角楽靖侍既考瑞藩劉須超黒多湘鵬然争郎洗慈搬。弓渡ゆ母郡出硫燃湾町鳳劇疎防-米曜宮短め決雲沖鄭統沃義仙き良病布梨敷々長清簗金仮概意岱桜潮率毒騎の法習運黄灤二下党底旗扉潭学福十生教貿積押勇履走招速視安舟遙位逢階蕭郷可映粛後丸送保華井面務島級串渓章登歳阪光反'


@register_charset('ja')
class JapaneseSet(BaseCharset):
    _CHARS = ja


@register_charset('de')
class DeCharSet(BaseCharset):

    _CHARS = u'abcdefghijklmnopqrstuvwxyzäöüß'
    _FEATURES = ['capitalization', 'umlaut']
    _UMLAUT = (u'ä', u'ö', u'ü')

    def process(self, word):
        ret = [copy.deepcopy(Map(self._feat_dict)) for _ in range(len(word))]
        for (i, c) in enumerate(word):
            if c in self._char2id:
                ret[i].update({'char': c})
                if c in self.__class__._UMLAUT:
                    ret[i].update({'umlaut': True})
            else:
                c_lower = c.lower()
                if c_lower in self.__class__._UMLAUT:
                    ret[i].update({'umlaut': True})
                if c_lower in self._char2id:
                    ret[i].update({'char': c_lower})
                    ret[i].update({'capitalization': True})
                else:
                    ret[i].update({'char': ''})
        return ret
