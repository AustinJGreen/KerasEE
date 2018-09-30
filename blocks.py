def is_solid(id):
    return (9 <= id and id <= 97 or 122 <= id and id <= 217 or id >= 1001 and id <= 1499 or id >= 2000) and id != 83 and id != 77 and id != 1520

def is_action(id):
    actionBlocks = {
        0,
        1,
        2,
        3,
        1518,
        4,
        459,
        411,
        412,
        413,
        1519,
        414,
        460,
        6,
        7,
        8,
        408,
        409,
        410,
        26,
        27,
        28,
        1008,
        1009,
        1010,
        23,
        24,
        25,
        1005,
        1006,
        1007,
        100,
        101,
        5,
        114,
        116,
        115,
        117,
        118,
        1534,
        120,
        98,
        99,
        424,
        472,
        361,
        1580,
        368,
        119,
        416,
        369,
        1064
    }

    return id in actionBlocks

def is_decoration(id):
    decorBlocks = {
        1000,
        1501,
        1503,
        1504,
        1505,
        1508,
        1509,
        1511,
        1512,
        1513,
        1514,
        1515,
        1516,
        1521,
        1522,
        1523,
        1524,
        1525,
        1526,
        1527,
        1528,
        1529,
        1530,
        1531,
        1532,
        1533,
        1534,
        1539,
        218,
        219,
        220,
        221,
        222,
        223,
        224,
        225,
        226,
        227,
        228,
        229,
        230,
        231,
        232,
        233,
        234,
        235,
        236,
        237,
        238,
        239,
        240,
        241,
        244,
        245,
        246,
        247,
        248,
        249,
        250,
        251,
        252,
        253,
        254,
        255,
        256,
        257,
        258,
        259,
        260,
        261,
        262,
        263,
        264,
        265,
        266,
        267,
        268,
        269,
        270,
        271,
        272,
        274,
        278,
        281,
        282,
        283,
        284,
        285,
        286,
        287,
        288,
        289,
        290,
        291,
        292,
        293,
        294,
        295,
        296,
        297,
        298,
        299,
        301,
        302,
        303,
        304,
        305,
        306,
        307,
        308,
        309,
        310,
        311,
        312,
        313,
        314,
        315,
        316,
        317,
        318,
        319,
        320,
        321,
        322,
        323,
        324,
        325,
        326,
        330,
        331,
        332,
        333,
        334,
        335,
        336,
        341,
        342,
        343,
        344,
        345,
        346,
        347,
        348,
        349,
        350,
        351,
        352,
        353,
        354,
        355,
        356,
        357,
        358,
        359,
        362,
        363,
        364,
        365,
        366,
        367,
        371,
        372,
        373,
        382,
        383,
        384,
        386,
        387,
        388,
        389,
        390,
        391,
        392,
        393,
        394,
        395,
        396,
        398,
        399,
        400,
        401,
        402,
        403,
        404,
        405,
        406,
        407,
        415,
        424,
        425,
        426,
        427,
        428,
        429,
        430,
        431,
        432,
        433,
        434,
        435,
        436,
        437,
        441,
        442,
        443,
        444,
        445,
        446,
        454,
        455,
        462,
        463,
        466,
        468,
        469,
        470,
        473,
        474,
        478,
        479,
        480,
        484,
        485,
        486,
        487,
        488,
        489,
        490,
        491,
        495,
        496
    }

    return id in decorBlocks

def simplify_block(id):
    convertMap = {
        411 : 1,
        412 : 2,
        413 : 3,
        1519 : 1518,
        414 : 4,
        460 : 459,
        1534 : 118,
        120 : 118,
        98 : 118,
        99 : 118,
        424 : 118,
        472 : 118,
        1146 : 118,
        1563 : 118,
        1580 : 361,
        368 : 361
    }

    if id in convertMap:
        return convertMap[id]
    elif is_solid(id):
        return 9
    else:
        return id

def rotate_block(id):
    convertMap = {
        0 : 1,
        1 : 2,
        2 : 3,
        3 : 0,
        411 : 412,
        412 : 413,
        413 : 0,
        117 : 114,
        114 : 116,
        116 : 115,
        115 : 117
    }

    if id in convertMap:
        return convertMap[id]
    else:
        return id