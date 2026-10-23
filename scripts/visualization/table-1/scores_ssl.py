import numpy as np
import tabulate
from scipy import stats

exp001 = {
    "exp001b": {
        999: {
            -1: [
                0.7438659603977987,
                0.7692233160386135,
                0.7647762657338184,
                0.7689443346178186,
                0.7731215457252315,
                0.7507920386795746,
                0.758600757907977,
                0.7580759997539998,
                0.7505841290364349,
                0.793301550809097,
                0.7427881947300838,
                0.7592010934550568,
                0.7621340549380484,
                0.7691761675365282,
                0.783249933900416,
            ]
        }
    },
    "exp001c": {
        999: {
            -1: [
                0.7858671945232755,
                0.8074114426454726,
                0.8043420441211996,
                0.8187558876300323,
                0.8227852897133262,
                0.7908025080629839,
                0.7857200070601353,
                0.8011818917437662,
                0.8220232744941705,
                0.8348634992203803,
                0.7844105305672268,
                0.8048126921980323,
                0.8071348074821808,
                0.8065053781737683,
                0.8235424695209806,
            ]
        }
    },
}

exp004 = {
    "exp004b": {
        1: {
            50: [
                0.464567877390447,
                0.48821059771562486,
                0.4252420294420059,
                0.483861437043276,
                0.476036986950218,
                0.4052140017793663,
                0.40667117717504314,
                0.43268919799264677,
                0.3682747532299269,
                0.2681078808824207,
                0.43766243413825673,
                0.5173761952295839,
                0.42888782318622665,
                0.31446382706503667,
                0.3760999268325814,
            ]
        }
    },
    "exp004c": {
        1: {
            50: [
                0.5060085815773949,
                0.5056418926224895,
                0.3946328137776875,
                0.48063779307059773,
                0.5629957223258958,
                0.39874348982281527,
                0.38248588243152515,
                0.47067448783434385,
                0.41223473479402706,
                0.2824412270858305,
                0.49242350251166744,
                0.5918758139443487,
                0.49641938481311065,
                0.31884366988861634,
                0.35943486058475366,
            ],
        },
    },
}

exp008 = {
    "exp008a": {
        1: {
            50: [
                0.4667707221472912,
                0.5035848080410016,
                0.4617004851628791,
                0.49338259783979843,
                0.5905519111104264,
                0.34869802998506827,
                0.37359408189928783,
                0.5197574662463469,
                0.34712701112845146,
                0.30478117902203133,
                0.4732221864437557,
                0.45910208668261443,
                0.4531194425770744,
                0.3829951728428349,
                0.35505942150890507,
            ]
        },
        0: {
            -1: [
                0.7400293883667375,
                0.75985508769042,
                0.7562590273801671,
                0.7776571565415189,
                0.8007025789155351,
                0.7359210789466665,
                0.7636062610028597,
                0.7494340990747357,
                0.7574693338705938,
                0.7902694850279465,
                0.7421438498043083,
                0.7592671140008024,
                0.7569835246380162,
                0.7507598279525964,
                0.7948621021499443,
            ]
        },
    },
    "exp008b": {
        1: {
            50: [
                0.49895811056539374,
                0.5148117197264961,
                0.48140020279388474,
                0.44642216263234635,
                0.5915228894134994,
                0.3407118447720104,
                0.33163958658672216,
                0.5189309065356543,
                0.37651451652160517,
                0.2759145355973451,
                0.4830696305365584,
                0.3876198401031659,
                0.47365317988552186,
                0.30646465119293553,
                0.3643640945804222,
            ]
        },
        0: {
            -1: [
                0.7922564357196633,
                0.8040512313712419,
                0.802655000988222,
                0.8112330447743981,
                0.8367995864351137,
                0.8040756252500982,
                0.778014849534358,
                0.7918758035235004,
                0.8028043804232778,
                0.833799415929302,
                0.8049895643111808,
                0.8087714966730107,
                0.787754496524474,
                0.8020839409650418,
                0.8380908851062193,
            ]
        },
    },
}

exp009 = {
    "exp009a": {
        1: {
            50: [
                0.4339238959727726,
                0.5652175113801933,
                0.45492141024832744,
                0.4753637891786065,
                0.6037051046805013,
                0.37705818058870266,
                0.425801989488909,
                0.5051331049091656,
                0.32035079562231444,
                0.30246804761872303,
                0.48093688243467725,
                0.5044244762634424,
                0.4914559970366822,
                0.39483954841925034,
                0.35342985935859855,
            ]
        },
        0: {
            -1: [
                0.7446602785619973,
                0.7703130871087251,
                0.755465339994125,
                0.7749537260319954,
                0.7836690351792408,
                0.7578590195112251,
                0.7651166880361502,
                0.7346358048090936,
                0.7606008055601872,
                0.8053327111259909,
                0.7209078492915626,
                0.7702658365950042,
                0.7447864907099981,
                0.7740199733768677,
                0.7998862731826901,
            ]
        },
    },
    "exp009b": {
        1: {
            50: [
                0.5420271110451722,
                0.5038913524686983,
                0.4604984930204922,
                0.44479228932124626,
                0.6036457067989569,
                0.3878222008216427,
                0.34132044129356204,
                0.5121182696323172,
                0.38827655960437957,
                0.27497490942079167,
                0.5526378694527042,
                0.4243669030401437,
                0.534302123587792,
                0.3582250882378261,
                0.37396524866500436,
            ]
        },
        0: {
            -1: [
                0.7869579812692632,
                0.8029618161224553,
                0.8106477697661187,
                0.8071301913963833,
                0.827626926431288,
                0.7965989451908897,
                0.8086741533835768,
                0.8069848413190357,
                0.8068884468491367,
                0.8314646091903353,
                0.7946334532286093,
                0.8095149332267424,
                0.8093333942837828,
                0.8278429082556042,
                0.8234739640130577,
            ]
        },
    },
}

exp010 = {
    "exp010a": {
        1: {
            50: [
                0.42541354698130396,
                0.4673145354731874,
                0.4353638990251576,
                0.48289767082660573,
                0.5723247284501796,
                0.36123014072005194,
                0.3762030391772657,
                0.4872524429009483,
                0.3357137229660109,
                0.27961249092318985,
                0.46067764456628846,
                0.49338491650158717,
                0.4698890740444829,
                0.3033417365856114,
                0.37338413138147775,
            ]
        },
        0: {
            -1: [
                0.7343223106228353,
                0.7729323463344252,
                0.7497456412772954,
                0.7563907111054801,
                0.7950634269114499,
                0.70640910681774,
                0.7610447249706709,
                0.7577914068995822,
                0.7498892801919936,
                0.7938022777410608,
                0.7162789655885344,
                0.7492536795527694,
                0.7482663574895776,
                0.7586137454135693,
                0.7892686472477646,
            ]
        },
    },
    "exp010b": {
        1: {
            50: [
                0.5470411809325896,
                0.5344716141737038,
                0.47641455758072715,
                0.49131243152527854,
                0.5266899877433884,
                0.2818027699649492,
                0.3914219210489905,
                0.5369602594064247,
                0.4007100060891455,
                0.2853618970778239,
                0.4949300325146817,
                0.5146229482980573,
                0.4497310517669061,
                0.3383252950206157,
                0.40141383779624035,
            ]
        },
        0: {
            -1: [
                0.7943793590837436,
                0.7898742179251519,
                0.7835640394978348,
                0.7904051361353642,
                0.8427926785590151,
                0.7845731855955131,
                0.7953849118869439,
                0.80064187308632,
                0.7892138074890631,
                0.8236866706311098,
                0.7885345157252883,
                0.7870708283350166,
                0.7754659075252904,
                0.7794857610041449,
                0.8331180622283558,
            ]
        },
    },
}

exp011 = {
    "exp011a": {
        1: {
            50: [
                0.43779658967331486,
                0.3890619466886339,
                0.4364974027065475,
                0.44517658862225423,
                0.5806691810609449,
                0.37408322397391247,
                0.358326637923425,
                0.4722615771225051,
                0.3562092626289438,
                0.2844914012609877,
                0.5075043944514792,
                0.4840120042341541,
                0.47572468980690363,
                0.3925463957781522,
                0.38312758731996477,
            ]
        },
        0: {
            -1: [
                0.7094492592999325,
                0.7568214874424702,
                0.7598973619226094,
                0.7722244275804535,
                0.7621190644145622,
                0.7381281167750402,
                0.7692828274235219,
                0.7596985280910967,
                0.7809632429220593,
                0.7723598828168204,
                0.7319561224189064,
                0.7524447549916459,
                0.7584322829843688,
                0.7669076874458673,
                0.8075532181641483,
            ]
        },
    },
    "exp011b": {
        1: {
            50: [
                0.5369873506803211,
                0.49255784393089314,
                0.46566959743470415,
                0.462350275868594,
                0.49223119137405663,
                0.3297408371000642,
                0.4091302530908905,
                0.5432316496245777,
                0.41500221714194907,
                0.27674721945887387,
                0.46232000922626815,
                0.5235794659909823,
                0.45945028631901874,
                0.32640024920766053,
                0.40723387218940416,
            ]
        },
        0: {
            -1: [
                0.7903529016054462,
                0.797731570627616,
                0.7979355395655444,
                0.7966784610791208,
                0.8274642090548443,
                0.7906069410017192,
                0.7978617710116671,
                0.8010249101534088,
                0.8077919988564458,
                0.8388418929263219,
                0.7923009854724076,
                0.8020250383730229,
                0.785846858140255,
                0.8089380756087661,
                0.8310025837335033,
            ]
        },
    },
}


if __name__ == "__main__":
    # columns should be: exp, low-data, high-data
    low_data_nsubj = 1
    high_data_nsubj_1 = 0
    high_data_nsubj_2 = 999
    low_data_nepochs = 50
    high_data_nepochs = -1

    # exp008
    simclr_eeg = [
        "SimCLR, pretrained with EEG data",
        f"{np.mean(exp008['exp008a'][low_data_nsubj][low_data_nepochs]):.2f} ({np.std(exp008['exp008a'][low_data_nsubj][low_data_nepochs]):.2f}) \\textsubscript{{$p={stats.wilcoxon(exp008['exp008a'][low_data_nsubj][low_data_nepochs], exp004['exp004b'][low_data_nsubj][low_data_nepochs]).pvalue:.2f}$}}",
        f"{np.mean(exp008['exp008b'][low_data_nsubj][low_data_nepochs]):.2f} ({np.std(exp008['exp008b'][low_data_nsubj][low_data_nepochs]):.2f}) \\textsubscript{{$p={stats.wilcoxon(exp008['exp008b'][low_data_nsubj][low_data_nepochs], exp004['exp004c'][low_data_nsubj][low_data_nepochs]).pvalue:.2f}$}}",
        f"{np.mean(exp008['exp008a'][high_data_nsubj_1][high_data_nepochs]):.2f} ({np.std(exp008['exp008a'][high_data_nsubj_1][high_data_nepochs]):.2f}) \\textsubscript{{$p={stats.wilcoxon(exp008['exp008a'][high_data_nsubj_1][high_data_nepochs], exp001['exp001b'][high_data_nsubj_2][high_data_nepochs]).pvalue:.2f}$}}",
        f"{np.mean(exp008['exp008b'][high_data_nsubj_1][high_data_nepochs]):.2f} ({np.std(exp008['exp008b'][high_data_nsubj_1][high_data_nepochs]):.2f}) \\textsubscript{{$p={stats.wilcoxon(exp008['exp008b'][high_data_nsubj_1][high_data_nepochs], exp001['exp001c'][high_data_nsubj_2][high_data_nepochs]).pvalue:.2f}$}}",
    ]

    # exp009
    vicreg_eeg = [
        "VICReg, pretrained with EEG data",
        f"{np.mean(exp009['exp009a'][low_data_nsubj][low_data_nepochs]):.2f} ({np.std(exp009['exp009a'][low_data_nsubj][low_data_nepochs]):.2f}) \\textsubscript{{$p={stats.wilcoxon(exp009['exp009a'][low_data_nsubj][low_data_nepochs], exp004['exp004b'][low_data_nsubj][low_data_nepochs]).pvalue:.2f}$}}",
        f"{np.mean(exp009['exp009b'][low_data_nsubj][low_data_nepochs]):.2f} ({np.std(exp009['exp009b'][low_data_nsubj][low_data_nepochs]):.2f}) \\textsubscript{{$p={stats.wilcoxon(exp009['exp009b'][low_data_nsubj][low_data_nepochs], exp004['exp004c'][low_data_nsubj][low_data_nepochs]).pvalue:.2f}$}}",
        f"{np.mean(exp009['exp009a'][high_data_nsubj_1][high_data_nepochs]):.2f} ({np.std(exp009['exp009a'][high_data_nsubj_1][high_data_nepochs]):.2f}) \\textsubscript{{$p={stats.wilcoxon(exp009['exp009a'][high_data_nsubj_1][high_data_nepochs], exp001['exp001b'][high_data_nsubj_2][high_data_nepochs]).pvalue:.2f}$}}",
        f"{np.mean(exp009['exp009b'][high_data_nsubj_1][high_data_nepochs]):.2f} ({np.std(exp009['exp009b'][high_data_nsubj_1][high_data_nepochs]):.2f}) \\textsubscript{{$p={stats.wilcoxon(exp009['exp009b'][high_data_nsubj_1][high_data_nepochs], exp001['exp001c'][high_data_nsubj_2][high_data_nepochs]).pvalue:.2f}$}}",
    ]

    # exp010
    simclr_synth = [
        "SimCLR, pretrained with synth. data",
        f"{np.mean(exp010['exp010a'][low_data_nsubj][low_data_nepochs]):.2f} ({np.std(exp010['exp010a'][low_data_nsubj][low_data_nepochs]):.2f}) \\textsubscript{{$p={stats.wilcoxon(exp010['exp010a'][low_data_nsubj][low_data_nepochs], exp004['exp004b'][low_data_nsubj][low_data_nepochs]).pvalue:.2f}$}}",
        f"{np.mean(exp010['exp010b'][low_data_nsubj][low_data_nepochs]):.2f} ({np.std(exp010['exp010b'][low_data_nsubj][low_data_nepochs]):.2f}) \\textsubscript{{$p={stats.wilcoxon(exp010['exp010b'][low_data_nsubj][low_data_nepochs], exp004['exp004c'][low_data_nsubj][low_data_nepochs]).pvalue:.2f}$}}",
        f"{np.mean(exp010['exp010a'][high_data_nsubj_1][high_data_nepochs]):.2f} ({np.std(exp010['exp010a'][high_data_nsubj_1][high_data_nepochs]):.2f}) \\textsubscript{{$p={stats.wilcoxon(exp010['exp010a'][high_data_nsubj_1][high_data_nepochs], exp001['exp001b'][high_data_nsubj_2][high_data_nepochs]).pvalue:.2f}$}}",
        f"{np.mean(exp010['exp010b'][high_data_nsubj_1][high_data_nepochs]):.2f} ({np.std(exp010['exp010b'][high_data_nsubj_1][high_data_nepochs]):.2f}) \\textsubscript{{$p={stats.wilcoxon(exp010['exp010b'][high_data_nsubj_1][high_data_nepochs], exp001['exp001c'][high_data_nsubj_2][high_data_nepochs]).pvalue:.2f}$}}",
    ]

    # exp011
    vicreg_synth = [
        "VICReg, pretrained with synth. data",
        f"{np.mean(exp011['exp011a'][low_data_nsubj][low_data_nepochs]):.2f} ({np.std(exp011['exp011a'][low_data_nsubj][low_data_nepochs]):.2f}) \\textsubscript{{$p={stats.wilcoxon(exp011['exp011a'][low_data_nsubj][low_data_nepochs], exp004['exp004b'][low_data_nsubj][low_data_nepochs]).pvalue:.2f}$}}",
        f"{np.mean(exp011['exp011b'][low_data_nsubj][low_data_nepochs]):.2f} ({np.std(exp011['exp011b'][low_data_nsubj][low_data_nepochs]):.2f}) \\textsubscript{{$p={stats.wilcoxon(exp011['exp011b'][low_data_nsubj][low_data_nepochs], exp004['exp004c'][low_data_nsubj][low_data_nepochs]).pvalue:.2f}$}}",
        f"{np.mean(exp011['exp011a'][high_data_nsubj_1][high_data_nepochs]):.2f} ({np.std(exp011['exp011a'][high_data_nsubj_1][high_data_nepochs]):.2f}) \\textsubscript{{$p={stats.wilcoxon(exp011['exp011a'][high_data_nsubj_1][high_data_nepochs], exp001['exp001b'][high_data_nsubj_2][high_data_nepochs]).pvalue:.2f}$}}",
        f"{np.mean(exp011['exp011b'][high_data_nsubj_1][high_data_nepochs]):.2f} ({np.std(exp011['exp011b'][high_data_nsubj_1][high_data_nepochs]):.2f}) \\textsubscript{{$p={stats.wilcoxon(exp011['exp011b'][high_data_nsubj_1][high_data_nepochs], exp001['exp001c'][high_data_nsubj_2][high_data_nepochs]).pvalue:.2f}$}}",
    ]

    fpt = [
        "Frequency Pretraining",
        f"{np.mean(exp004['exp004b'][low_data_nsubj][low_data_nepochs]):.2f} ({np.std(exp004['exp004b'][low_data_nsubj][low_data_nepochs]):.2f})",
        f"{np.mean(exp004['exp004c'][low_data_nsubj][low_data_nepochs]):.2f} ({np.std(exp004['exp004c'][low_data_nsubj][low_data_nepochs]):.2f})",
        f"{np.mean(exp001['exp001b'][high_data_nsubj_2][high_data_nepochs]):.2f} ({np.std(exp001['exp001b'][high_data_nsubj_2][high_data_nepochs]):.2f})",
        f"{np.mean(exp001['exp001c'][high_data_nsubj_2][high_data_nepochs]):.2f} ({np.std(exp001['exp001c'][high_data_nsubj_2][high_data_nepochs]):.2f})",
    ]

    table = tabulate.tabulate(
        [
            simclr_synth,
            simclr_eeg,
            vicreg_synth,
            vicreg_eeg,
            fpt,
        ],
        headers=[
            "exp",
            "low-data, fixed fe",
            "low-data, fine-tuned FT",
            "high-data, fixed fe",
            "high-data, fine-tuned FT",
        ],
        tablefmt="latex_raw",  # latex_raw
        floatfmt=".2f",
    )
    print(table)
