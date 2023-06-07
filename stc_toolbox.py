import random
import math
import time

payload = 0.5
n = int(1e+5)
h = 10
m = math.floor(payload * n)
trials = 10

def mod(x, m):
    return (x - (x // m) * m + m) % m

def _MM_SHUFFLE(w, x, y, z):
    return (w << 6) | (x << 4) | (y << 2) | z

def _mm_shuffle_ps(a, b, control):
    def select4(src, control):
        if control == 0:
            return src & 0xFFFFFFFF
        elif control == 1:
            return (src & (0xFFFFFFFF << 32)) >> 32
        elif control == 2:
            return (src & (0xFFFFFFFF << 64)) >> 64
        elif control == 3:
            return (src & (0xFFFFFFFF << 96)) >> 96

    return (select4(b, control >> 6) << 96) | (select4(b, (control >> 4) & 3) << 64) \
        | (select4(a, (control >> 2) & 3) << 32) | select4(a, control & 3)

def sum_inplace(x):
    x += _mm_shuffle_ps(x,x,_MM_SHUFFLE(1,0,3,2))
    x += _mm_shuffle_ps(x,x,_MM_SHUFFLE(2,3,0,1))
    y = x
    return y

def calc_entropy(n, k, costs, lambda_ ):
    LOG2 = math.log(2.0, math.e)
    inf = F_INF
    v_lambda = -lambda_
    entr = 0;
    for i in range(n / 4):
        z = 0
        d = 0
        for j in range (k):
            rho = costs + j * n + 4 * i
            p = int(math.exp( v_lambda * rho))
            z += p

            mask = rho == inf
            p = rho * p
            p = mask & (~p)
            d += p
        entr = entr -  ((v_lambda * d)  // z)
        entr = entr + math.log(z, 2)
    return sum_inplace( entr ) / LOG2

def check_costs(n, k, costs):
    for i in range(n):
        test_nan, test_non_inf, test_minus_inf = False, False, False
        for j in range(k):
            test_nan |= math.isnan(costs[k * i + j])
            test_non_inf |= ((not math.isinf(costs[k * i + j])) & (not math.isneginf(costs[k * i + j])))
            test_minus_inf |= (costs[k * i + j] == float('-inf'))
        if test_nan:
            message = f"Incorrect cost array.{i}-th element contains NaN value. This is not a valid cost."
            raise ValueError(message)
        if not test_non_inf:
            message = f"Incorrect cost array.{i}-th element does not contain any finite cost value. This is not a valid cost."
            raise ValueError(message)
        if test_minus_inf:
            message = f"Incorrect cost array.{i}-th element contains -Inf value. This is not a valid cost."
            raise ValueError(message)


def random_permutation(n, seed, perm):
    random.seed(seed)

    # generate random permutation
    for i in range(n):
        perm[i] = i
    for i in range(n):
        j = random.randint(0, n - i - 1)
        tmp = perm[i]
        perm[i] = perm[i + j]
        perm[i + j] = tmp
    return perm

def calc_distortion(n, k, costs, lambda_):
    eps = np.finfo(np.float32).eps
    v_lambda = -lambda_

    dist = 0
    for i in range(0, n, 4):
        z = 0
        d = 0
        for j in range(k):
            rho = costs[j*n + i:j*n + i + 4]
            p = math.exp(v_lambda * rho)
            z += p
            mask = p < eps
            p = rho * p
            p = mask & (~p)
            d += p
        dist += d // z
    return np.sum(dist)

def get_lambda_distortion(n, k, costs, distortion, initial_lambda=10, precision=1e-3, iter_limit=30):
    j = 0
    iterations = 0
    lambda1 = 0
    dist1 = calc_distortion(n, k, costs, lambda1)
    lambda3 = initial_lambda
    dist2 = float('inf')
    lambda2 = initial_lambda
    dist3 = distortion + 1
    while dist3 > distortion:
        lambda3 *= 2
        dist3 = calc_distortion(n, k, costs, lambda3)
        j += 1
        iterations += 1
        if j > 10:
            return lambda3
    while abs(dist2 - distortion) / n > precision and iterations < iter_limit:
        lambda2 = lambda1 + (lambda3 - lambda1) / 2
        dist2 = calc_distortion(n, k, costs, lambda2)
        if dist2 < distortion:
            lambda3 = lambda2
            dist3 = dist2
        else:
            lambda1 = lambda2
            dist1 = dist2
        iterations += 1
    return lambda1 + (lambda3 - lambda1) / 2

def stc_embed(vector, vectorlength, syndrome, syndromelength, pricevectorv, usefloat,
        stego, matrixheight ):
    # height, i, k, l, index, index2, parts, m, sseheight, altm, pathindex: int
    # column, colmask, state: int
    totalprice: float

    ssedone: int
    path = [0]
    columns = [0] * 2
    matrices = []
    widths = []

    if matrixheight > 31:
        raise ValueError( "Submatrix height must not exceed 31.", 1 )

    height = 1 << matrixheight
    colmask = height - 1
    height = (height + 31) & (~31)

    parts = height >> 5

    if stego is not None:
        path = [0] * (vectorlength * parts * 4);
        if path is None:
            error_msg = "Not enough memory (" + (vectorlength * parts * 4) + " byte array could not be allocated)."
            raise ValueError( error_msg, 2 )
        pathindex = 0

        # int shorter, longer, worm;
        # double invalpha;

        matrices = [0] * syndromelength
        widths = [0] * syndromelength

        invalpha = vectorlength / syndromelength
        if invalpha < 1:
            raise ValueError( "The message cannot be longer than the cover object.", 3 )
        """
         THIS IS OBSOLETE. Algorithm still works for alpha >1/2. You need to take care of cases with too many Infs in cost vector.
         if(invalpha < 2) {
         printf("The relative payload is greater than 1/2. This may result in poor embedding efficiency.\n");
         }
         """
        shorter = int(invalpha)
        longer = math.ceil( invalpha )
        columns[0] = getMatrix(shorter, matrixheight)
        if columns[0] is None:
            return -1

        columns[1] = getMatrix(longer, matrixheight)
        if columns[1] is None:
            return -1
        worm = 0
        for i in range(syndromelength):
            if worm + longer <= (i + 1) * invalpha + 0.5:
                matrices[i] = 1
                widths[i] = longer
                worm += longer
            else:
                matrices[i] = 0
                widths[i] = shorter
                worm += shorter
    if usefloat:
        /*
         SSE FLOAT VERSION
         */
        pathindex8 = 0
        shift = [ 0, 4 ]
        mask = [ 0xf0, 0x0f ]
        prices = []
        path8 = path
        pricevector = pricevectorv
        total = 0
        inf = F_INF

        sseheight = height >> 2;
        ssedone = [0] * sseheight
        prices = [16] * height;

        fillval = inf
        for i in range(0, height, 4):
            prices[i] = fillval
            ssedone[i >> 2] = 0


        prices[0] = 0.0

        index = 0
        for index2 in range(syndromelength):
            c1 = 0
            c2 = 0

            for k in range(widths[index2]):
                column = columns[matrices[index2]][k] & colmask

                if vector[index] == 0:
                    c1 = 0
                    c2 = (float) pricevector[index]
                else:
                    c1 = pricevector[index]
                    c2 = 0

                total += pricevector[index]

                for m in range(sseheight):
                    if not ssedone[m]:
                        altm = (m ^ (column >> 2))
                        v1 = prices[m << 2]
                        v2 = prices[altm << 2]
                        v3 = v1;
                        v4 = v2;
                        ssedone[m] = 1;
                        ssedone[altm] = 1;
                        last_bits = column & 3
                        if last_bits == 0:
                            break;
                        elif last_bits == 1:
                                v2 = _mm_shuffle_ps(v2, v2, 0xb1);
                                v3 = _mm_shuffle_ps(v3, v3, 0xb1);
                                break;
                        elif last_bits == 2:
                                v2 = _mm_shuffle_ps(v2, v2, 0x4e);
                                v3 = _mm_shuffle_ps(v3, v3, 0x4e);
                                break;
                        elif last_bits == 3:
                                v2 = _mm_shuffle_ps(v2, v2, 0x1b);
                                v3 = _mm_shuffle_ps(v3, v3, 0x1b);
                                break
                        v1 += c1
                        v2 += c2
                        v3 += c2
                        v4 += c1

                        v1 = min( v1, v2 )
                        v4 = min( v3, v4 )

                        prices[m << 2] = v1
                        prices[altm << 2] = v4

                        if stego is not None:
                            v2 = v1 == v2
                            v3 = v3 == v4
                            path8[pathindex8 + (m >> 1)] = (path8[pathindex8 + (m >> 1)] & mask[m & 1]) | (_mm_movemask_ps( v2 ) << shift[m
                                    & 1]);
                            path8[pathindex8 + (altm >> 1)] = (path8[pathindex8 + (altm >> 1)] & mask[altm & 1]) | (_mm_movemask_ps( v3 )
                                    << shift[altm & 1])

                for i in range(sseheight):
                    ssedone[i] = 0

                pathindex += parts
                pathindex8 += parts << 2;
                index += 1

            if syndrome[index2] == 0:
                l = 0
                for i in range(0, sseheight, 2):
                    prices[l] = _mm_shuffle_ps(prices[i << 2], prices[(i + 1) << 2], 0x88)
                    l += 4
            else:
                l = 0
                for i in range(0, sseheight, 2):
                    prices[l] = _mm_shuffle_ps(prices[i << 2], prices[(i + 1) << 2], 0xdd)
                    l += 4

            if syndromelength - index2 <= matrixheight:
                colmask >>= 1;

            fillval = inf
            for i in range(l >> 2, sseheight):
                &prices[l << 2] = fillval

        totalprice = prices[0]

        if totalprice >= total:
            raise ValueError( "No solution exist.", 4 )
    else:
        /*
         SSE UINT8 VERSION
         */
        int pathindex16 = 0, subprice = 0;
        u8 maxc = 0, minc = 0;
        u8 *prices, *pricevector = (u8*) pricevectorv;
        u16 *path16 = (u16 *) path;
        __m128i *prices16B;

        sseheight = height >> 4;
        ssedone = (u8*) malloc( sseheight * sizeof(u8) );
        prices = (u8*) aligned_malloc( height * sizeof(u8), 16 );
        prices16B = (__m128i *) prices;

        {
            __m128i napln = _mm_set1_epi32( 0xffffffff );
            for ( i = 0; i < sseheight; i++ ) {
                _mm_store_si128( &prices16B[i], napln );
                ssedone[i] = 0;
            }
        }

        prices[0] = 0;

        for ( index = 0, index2 = 0; index2 < syndromelength; index2++ ) {
            register __m128i c1, c2, maxp, minp;

            if ( (u32) maxc + pricevector[index] >= 254 ) {
                aligned_free( path );
                free( ssedone );
                free( matrices );
                free( widths );
                free( columns[0] );
                free( columns[1] );
                if ( stego != NULL ) free( path );
                throw stc_exception( "Price vector limit exceeded.", 5 );
            }

            for ( k = 0; k < widths[index2]; k++, index++ ) {
                column = columns[matrices[index2]][k] & colmask;

                if ( vector[index] == 0 ) {
                    c1 = _mm_setzero_si128();
                    c2 = _mm_set1_epi8( pricevector[index] );
                } else {
                    c1 = _mm_set1_epi8( pricevector[index] );
                    c2 = _mm_setzero_si128();
                }

                minp = _mm_set1_epi8( -1 );
                maxp = _mm_setzero_si128();

                for ( m = 0; m < sseheight; m++ ) {
                    if ( !ssedone[m] ) {
                        register __m128i v1, v2, v3, v4;
                        altm = (m ^ (column >> 4));
                        v1 = _mm_load_si128( &prices16B[m] );
                        v2 = _mm_load_si128( &prices16B[altm] );
                        v3 = v1;
                        v4 = v2;
                        ssedone[m] = 1;
                        ssedone[altm] = 1;
                        if ( column & 8 ) {
                            v2 = _mm_shuffle_epi32(v2, 0x4e);
                            v3 = _mm_shuffle_epi32(v3, 0x4e);
                        }
                        if ( column & 4 ) {
                            v2 = _mm_shuffle_epi32(v2, 0xb1);
                            v3 = _mm_shuffle_epi32(v3, 0xb1);
                        }
                        if ( column & 2 ) {
                            v2 = _mm_shufflehi_epi16(v2, 0xb1);
                            v3 = _mm_shufflehi_epi16(v3, 0xb1);
                            v2 = _mm_shufflelo_epi16(v2, 0xb1);
                            v3 = _mm_shufflelo_epi16(v3, 0xb1);
                        }
                        if ( column & 1 ) {
                            v2 = _mm_or_si128( _mm_srli_epi16( v2, 8 ), _mm_slli_epi16( v2, 8 ) );
                            v3 = _mm_or_si128( _mm_srli_epi16( v3, 8 ), _mm_slli_epi16( v3, 8 ) );
                        }
                        v1 = _mm_adds_epu8( v1, c1 );
                        v2 = _mm_adds_epu8( v2, c2 );
                        v3 = _mm_adds_epu8( v3, c2 );
                        v4 = _mm_adds_epu8( v4, c1 );

                        v1 = _mm_min_epu8( v1, v2 );
                        v4 = _mm_min_epu8( v3, v4 );

                        _mm_store_si128( &prices16B[m], v1 );
                        _mm_store_si128( &prices16B[altm], v4 );

                        minp = _mm_min_epu8( minp, _mm_min_epu8( v1, v4 ) );
                        maxp = _mm_max_epu8( maxp, maxLessThan255( v1, v4 ) );

                        if ( stego != NULL ) {
                            v2 = _mm_cmpeq_epi8( v1, v2 );
                            v3 = _mm_cmpeq_epi8( v3, v4 );
                            path16[pathindex16 + m] = (u16) _mm_movemask_epi8( v2 );
                            path16[pathindex16 + altm] = (u16) _mm_movemask_epi8( v3 );
                        }
                    }
                }

                maxc = max16B( maxp );
                minc = min16B( minp );

                maxc -= minc;
                subprice += minc;
                {
                    register __m128i mask = _mm_set1_epi32( 0xffffffff );
                    register __m128i m = _mm_set1_epi8( minc );
                    for ( i = 0; i < sseheight; i++ ) {
                        register __m128i res;
                        register __m128i pr = prices16B[i];
                        res = _mm_andnot_si128( _mm_cmpeq_epi8( pr, mask ), m );
                        prices16B[i] = _mm_sub_epi8( pr, res );
                        ssedone[i] = 0;
                    }
                }

                pathindex += parts;
                pathindex16 += parts << 1;
            }

            {
                register __m128i mask = _mm_set1_epi32( 0x00ff00ff );

                if ( minc == 255 ) {
                    aligned_free( path );
                    free( ssedone );
                    free( matrices );
                    free( widths );
                    free( columns[0] );
                    free( columns[1] );
                    if ( stego != NULL ) free( path );
                    throw stc_exception( "The syndrome is not in the syndrome matrix range.", 4 );
                }

                if ( syndrome[index2] == 0 ) {
                    for ( i = 0, l = 0; i < sseheight; i += 2, l++ ) {
                        _mm_store_si128( &prices16B[l], _mm_packus_epi16( _mm_and_si128( _mm_load_si128( &prices16B[i] ), mask ),
                                _mm_and_si128( _mm_load_si128( &prices16B[i + 1] ), mask ) ) );
                    }
                } else {
                    for ( i = 0, l = 0; i < sseheight; i += 2, l++ ) {
                        _mm_store_si128( &prices16B[l], _mm_packus_epi16( _mm_and_si128( _mm_srli_si128(_mm_load_si128(&prices16B[i]), 1),
                                mask ), _mm_and_si128( _mm_srli_si128(_mm_load_si128(&prices16B[i + 1]), 1), mask ) ) );
                    }
                }

                if ( syndromelength - index2 <= matrixheight ) colmask >>= 1;

                register __m128i fillval = _mm_set1_epi32( 0xffffffff );
                for ( ; l < sseheight; l++ )
                    _mm_store_si128( &prices16B[l], fillval );
            }
        }

        totalprice = subprice + prices[0];

        aligned_free( prices );
        free( ssedone );
    }

    if ( stego != NULL ) {
        pathindex -= parts;
        index--;
        index2--;
        state = 0;

        // unused
        // int h = syndromelength;
        state = 0;
        colmask = 0;
        for ( ; index2 >= 0; index2-- ) {
            for ( k = widths[index2] - 1; k >= 0; k--, index-- ) {
                if ( k == widths[index2] - 1 ) {
                    state = (state << 1) | syndrome[index2];
                    if ( syndromelength - index2 <= matrixheight ) colmask = (colmask << 1) | 1;
                }

                if ( path[pathindex + (state >> 5)] & (1 << (state & 31)) ) {
                    stego[index] = 1;
                    state = state ^ (columns[matrices[index2]][k] & colmask);
                } else {
                    stego[index] = 0;
                }

                pathindex -= parts;
            }
        }
        free( path );
    }

    free( matrices );
    free( widths );
    free( columns[0] );
    free( columns[1] );

    return totalprice;

def stc_ml1_embed(cover_length, cover, direction, costs, message_length, message, target_distortion,
                  stc_constraint_height, expected_coding_loss, stego, num_msg_bits, max_trials, coding_loss):
    distortion = 0.0
    lambda_value = 0.0
    m_max = 0.0
    success = False
    m_actual = 0
    n = cover_length + 4 - (cover_length % 4)  # cover length rounded to multiple of 4
    perm1 = [0] * n

    c = [0.0] * (2 * n)
    for i in range(2 * n):
        c[i] = float("inf")
    for i in range(0, cover_length):
        c[mod(cover[i], 2) * n + i] = 0  # cost of not changing the element
        c[mod((cover[i] + 1), 2) * n + i] = costs[i]  # cost of changing the element

    if target_distortion != float("inf"):  # distortion-limited sender
        lambda_value = get_lambda_distortion(n, 2, c, target_distortion, 2)
        m_max = (1 - expected_coding_loss) * calc_entropy(n, 2, c, lambda_value)
        m_actual = min(message_length, int(m_max))
    if target_distortion == float("inf") or m_actual < int(m_max):  # payload-limited sender
        m_actual = min(cover_length, message_length)
    # SINGLE LAYER OF 1ST LSBs
    num_msg_bits[0] = m_actual
    trial = 0
    cover1 = bytearray(cover_length)
    cost1 = [0.0] * cover_length
    stego1 = bytearray(cover_length)
    while not success:
        perm1 = random_permutation(cover_length, 100, num_msg_bits[0])
        for i in range(cover_length):
            cover1[perm1[i]] = cover[i] % 2
            cost1[perm1[i]] = costs[i]
            if cost1[perm1[i]] != cost1[perm1[i]]:
                cost1[perm1[i]] = float('inf')
        stego1[:] = cover1[:]
        try:
            if num_msg_bits[0] != 0:
                stc_embed(cover1, cover_length, message, num_msg_bits[0], cost1, True,
                          stego1, stc_constraint_height)
                success = True
        except ValueError as e:
            if e.error_id != 4:
                raise e
            num_msg_bits[0] -= 1
            trial += 1
            if trial > max_trials:
                raise ValueError("Maximum number of trials in layered construction exceeded.", 6)


def stc_ml2_embed(cover_length, costs, stego_values, message_length, message, target_distortion,
                  stc_constraint_height, expected_coding_loss, stego, num_msg_bits, max_trials, coding_loss):
    distortion, dist_coding_loss, lambda_value = 0, 0, 0
    m_max, m_actual = 0, 0
    n = cover_length + 4 - (cover_length % 4)

    check_costs(cover_length, 4, costs)

    lsb1_only = True
    for i in range(cover_length):
        n_finite_costs = 0
        lsb_xor = 0
        for k in range(4):
            if not math.isinf(costs[4 * i + k]):
                n_finite_costs += 1
                lsb_xor ^= (k % 2)
        lsb1_only &= ((n_finite_costs <= 2) & (lsb_xor == 1))
    if lsb1_only:  # use stcml1embed method
        distortion = 0
        cover = [0] * cover_length
        direction = [0] * cover_length
        costs_ml1 = [0] * cover_length
        cover_length
        for i in range(cover_length):
            minid = 0
            fmin = float("inf")
            for j in range(4):
                if fmin > costs[4 * i + j]:
                    fmin = costs[4 * i + j]  # minimum value
                    minid = j  # index of the minimal entry
            costs_ml1[i] = float("inf")
            cover[i] = stego_values[4 * i + minid]
            for j in range(4):
                if (costs[4 * i + j] != float("inf")) and (minid != j):
                    distortion += fmin
                    costs_ml1[i] = costs[4 * i + j] - fmin
                    direction[i] = stego_values[4 * i + j] - cover[i]

        distortion += stc_ml1_embed(cover_length, cover, direction, costs_ml1, message_length, message,
                                    target_distortion, stc_constraint_height, expected_coding_loss, stego,
                                    num_msg_bits, max_trials, coding_loss)

        return distortion


def stc_pm1_dls_embed(cover_length, cover, costs, message_length, message, target_distortion, stc_constraint_height,
                      expected_coding_loss, wet_cost, stego, num_msg_bits, max_trials, coding_loss):
    check_costs(cover_length, 3, costs)
    dist = 0
    stego_values = [0] * (4 * cover_length)
    costs_ml2 = [0.0] * (4 * cover_length)
    for i in range(cover_length):
        costs_ml2[4 * i + ((cover[i] - 1 + 4) % 4)] = costs[3 * i + 0]
        stego_values[4 * i + ((cover[i] - 1 + 4) % 4)] = cover[i] - 1
        costs_ml2[4 * i + ((cover[i] + 0 + 4) % 4)] = costs[3 * i + 1]
        stego_values[4 * i + ((cover[i] + 0 + 4) % 4)] = cover[i]
        costs_ml2[4 * i + ((cover[i] + 1 + 4) % 4)] = costs[3 * i + 2]
        stego_values[4 * i + ((cover[i] + 1 + 4) % 4)] = cover[i] + 1
        costs_ml2[4 * i + ((cover[i] + 2 + 4) % 4)] = wet_cost
        stego_values[4 * i + ((cover[i] + 2 + 4) % 4)] = cover[i] + 2
    dist = stc_ml2_embed(cover_length, costs_ml2, stego_values, message_length, message, target_distortion,
                         stc_constraint_height, expected_coding_loss, stego, num_msg_bits, max_trials, coding_loss)
    return dist



random.seed(0)

cover = [random.randint(0, 255) for i in range(n)]
costs = [0] * (3 * n)
F_INF = float('inf')

for i in range(n):
    if cover[i] == 0:
        costs[3 * i + 0] = F_INF
        costs[3 * i + 1] = 0
        costs[3 * i + 2] = 1
    elif cover[i] == 255:
        costs[3 * i + 0] = 1
        costs[3 * i + 1] = 0
        costs[3 * i + 2] = F_INF
    else:
        costs[3 * i + 0] = 1
        costs[3 * i + 1] = 0
        costs[3 * i + 2] = 1

message = bytearray(m)

import numpy as np

extracted_message = np.zeros(m, dtype=np.uint8)
for i in range(m):
    extracted_message[i] = random.randint(0, 1)
stego = np.zeros(n, dtype=np.int32)
num_msg_bits = np.zeros(2, dtype=np.uint32)
coding_loss = 0.0
print("Multi layer construction for steganography.\nExample of weighted +-1 embedding using 2 layers of STCs.\n\n")
print("Running stc_pm1_pls_embed()    WITH coding loss calculation ... ", end="")
t0 = time.perf_counter()
dist = stc_pm1_pls_embed(n, cover, costs, m, extracted_message, h, F_INF, stego, num_msg_bits, trials, coding_loss)
t = time.perf_counter() - t0
print(f"done in {t} seconds.")

trials = 10
print("Running stc_pm1_pls_embed() WITHOUT coding loss calculation ... ", end="")
t0 = time.perf_counter()
dist = stc_pm1_pls_embed(n, cover, costs, m, message, h, F_INF, stego, num_msg_bits, trials, 0)
t = time.perf_counter() - t0
print(f"done in {t} seconds.")
print("Running stc_ml2_extract() ... ", end="")
t0 = time.perf_counter()
stc_ml_extract(n, stego, 2, num_msg_bits, h, extracted_message)
print(f"done in {time.perf_counter() - t0} seconds.\n")

print(f"          Cover size  n = {n} elements.")
print(f"         Message bits m = {m} bits => {num_msg_bits[1]} bits in 2LSBs and {num_msg_bits[0]} bits in LSBs.")
print(f"STC constraint height h = {h} bits")
print(f"          Coding loss l = {coding_loss * 100}%.")
print(f"     Processing speed t = {n / t} cover elements/second without coding loss calculation")
msg_ok = True
for i in range(m):
    msg_ok &= (extracted_message[i] == message[i])
    if not msg_ok:
        print(f"\nExtracted message differs in bit {i}")
if msg_ok:
    print("\nMessage was embedded and extracted correctly.")
