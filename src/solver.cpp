#define _CRT_NONSTDC_NO_WARNINGS
#define _SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING
#include <bits/stdc++.h>
#include <random>
#include <unordered_set>
#include <array>
#ifdef _MSC_VER
#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <conio.h>
#include <ppl.h>
#include <filesystem>
#include <intrin.h>
#include <boost/multiprecision/cpp_int.hpp>
int __builtin_clz(unsigned int n)
{
    unsigned long index;
    _BitScanReverse(&index, n);
    return 31 - index;
}
int __builtin_ctz(unsigned int n)
{
    unsigned long index;
    _BitScanForward(&index, n);
    return index;
}
namespace std {
    inline int __lg(int __n) { return sizeof(int) * 8 - 1 - __builtin_clz(__n); }
}
using __uint128_t = boost::multiprecision::uint128_t;
#else
#pragma GCC target("avx2")
#pragma GCC optimize("O3")
#pragma GCC optimize("unroll-loops")
#endif

// hashset: https://nyaannyaan.github.io/library/hashmap/hashset.hpp
namespace HashMapImpl {

    using namespace std;

    using u32 = uint32_t;
    using u64 = uint64_t;

    template <typename Key, typename Data>
    struct HashMapBase;

    template <typename Key, typename Data>
    struct itrB
        : iterator<bidirectional_iterator_tag, Data, ptrdiff_t, Data*, Data&> {
        using base =
            iterator<bidirectional_iterator_tag, Data, ptrdiff_t, Data*, Data&>;
        using ptr = typename base::pointer;
        using ref = typename base::reference;

        u32 i;
        HashMapBase<Key, Data>* p;

        explicit constexpr itrB() : i(0), p(nullptr) {}
        explicit constexpr itrB(u32 _i, HashMapBase<Key, Data>* _p) : i(_i), p(_p) {}
        explicit constexpr itrB(u32 _i, const HashMapBase<Key, Data>* _p)
            : i(_i), p(const_cast<HashMapBase<Key, Data>*>(_p)) {}
        friend void swap(itrB& l, itrB& r) { swap(l.i, r.i), swap(l.p, r.p); }
        friend bool operator==(const itrB& l, const itrB& r) { return l.i == r.i; }
        friend bool operator!=(const itrB& l, const itrB& r) { return l.i != r.i; }
        const ref operator*() const {
            return const_cast<const HashMapBase<Key, Data>*>(p)->data[i];
        }
        ref operator*() { return p->data[i]; }
        ptr operator->() const { return &(p->data[i]); }

        itrB& operator++() {
            assert(i != p->cap && "itr::operator++()");
            do {
                i++;
                if (i == p->cap) break;
                if (p->flag[i] == true && p->dflag[i] == false) break;
            } while (true);
            return (*this);
        }
        itrB operator++(int) {
            itrB it(*this);
            ++(*this);
            return it;
        }
        itrB& operator--() {
            do {
                i--;
                if (p->flag[i] == true && p->dflag[i] == false) break;
                assert(i != 0 && "itr::operator--()");
            } while (true);
            return (*this);
        }
        itrB operator--(int) {
            itrB it(*this);
            --(*this);
            return it;
        }
    };

    template <typename Key, typename Data>
    struct HashMapBase {
        using u32 = uint32_t;
        using u64 = uint64_t;
        using iterator = itrB<Key, Data>;
        using itr = iterator;

    protected:
        template <typename K>
        inline u64 randomized(const K& key) const {
            return u64(key) ^ r;
        }

        template <typename K,
            enable_if_t<is_same<K, Key>::value, nullptr_t> = nullptr,
            enable_if_t<is_integral<K>::value, nullptr_t> = nullptr>
            inline u32 inner_hash(const K& key) const {
            return (randomized(key) * 11995408973635179863ULL) >> shift;
        }
        template <
            typename K, enable_if_t<is_same<K, Key>::value, nullptr_t> = nullptr,
            enable_if_t<is_integral<decltype(K::first)>::value, nullptr_t> = nullptr,
            enable_if_t<is_integral<decltype(K::second)>::value, nullptr_t> = nullptr>
            inline u32 inner_hash(const K& key) const {
            u64 a = randomized(key.first), b = randomized(key.second);
            a *= 11995408973635179863ULL;
            b *= 10150724397891781847ULL;
            return (a + b) >> shift;
        }
        template <typename K,
            enable_if_t<is_same<K, Key>::value, nullptr_t> = nullptr,
            enable_if_t<is_integral<typename K::value_type>::value, nullptr_t> =
            nullptr>
            inline u32 inner_hash(const K& key) const {
            static constexpr u64 mod = (1LL << 61) - 1;
            static constexpr u64 base = 950699498548472943ULL;
            u64 res = 0;
            for (auto& elem : key) {
                __uint128_t x = __uint128_t(res) * base + (randomized(elem) & mod);
                res = (x & mod) + (x >> 61);
            }
            __uint128_t x = __uint128_t(res) * base;
            res = (x & mod) + (x >> 61);
            if (res >= mod) res -= mod;
            return res >> (shift - 3);
        }

        template <typename D = Data,
            enable_if_t<is_same<D, Key>::value, nullptr_t> = nullptr>
            inline u32 hash(const D& dat) const {
            return inner_hash(dat);
        }
        template <
            typename D = Data,
            enable_if_t<is_same<decltype(D::first), Key>::value, nullptr_t> = nullptr>
            inline u32 hash(const D& dat) const {
            return inner_hash(dat.first);
        }

        template <typename D = Data,
            enable_if_t<is_same<D, Key>::value, nullptr_t> = nullptr>
            inline Key dtok(const D& dat) const {
            return dat;
        }
        template <
            typename D = Data,
            enable_if_t<is_same<decltype(D::first), Key>::value, nullptr_t> = nullptr>
            inline Key dtok(const D& dat) const {
            return dat.first;
        }

        void reallocate(u32 ncap) {
            vector<Data> ndata(ncap);
            vector<bool> nf(ncap);
            shift = 64 - __lg(ncap);
            for (u32 i = 0; i < cap; i++) {
                if (flag[i] == true && dflag[i] == false) {
                    u32 h = hash(data[i]);
                    while (nf[h]) h = (h + 1) & (ncap - 1);
                    ndata[h] = move(data[i]);
                    nf[h] = true;
                }
            }
            data.swap(ndata);
            flag.swap(nf);
            cap = ncap;
            dflag.resize(cap);
            fill(std::begin(dflag), std::end(dflag), false);
        }

        inline bool extend_rate(u32 x) const { return x * 2 >= cap; }

        inline bool shrink_rate(u32 x) const {
            return HASHMAP_DEFAULT_SIZE < cap&& x * 10 <= cap;
        }

        inline void extend() { reallocate(cap << 1); }

        inline void shrink() { reallocate(cap >> 1); }

    public:
        u32 cap, s;
        vector<Data> data;
        vector<bool> flag, dflag;
        u32 shift;
        static u64 r;
        static constexpr uint32_t HASHMAP_DEFAULT_SIZE = 4;

        explicit HashMapBase()
            : cap(HASHMAP_DEFAULT_SIZE),
            s(0),
            data(cap),
            flag(cap),
            dflag(cap),
            shift(64 - __lg(cap)) {}

        itr begin() const {
            u32 h = 0;
            while (h != cap) {
                if (flag[h] == true && dflag[h] == false) break;
                h++;
            }
            return itr(h, this);
        }
        itr end() const { return itr(this->cap, this); }

        friend itr begin(const HashMapBase& h) { return h.begin(); }
        friend itr end(const HashMapBase& h) { return h.end(); }

        itr find(const Key& key) const {
            u32 h = inner_hash(key);
            while (true) {
                if (flag[h] == false) return this->end();
                if (dtok(data[h]) == key) {
                    if (dflag[h] == true) return this->end();
                    return itr(h, this);
                }
                h = (h + 1) & (cap - 1);
            }
        }

        bool contain(const Key& key) const { return find(key) != this->end(); }

        itr insert(const Data& d) {
            u32 h = hash(d);
            while (true) {
                if (flag[h] == false) {
                    if (extend_rate(s + 1)) {
                        extend();
                        h = hash(d);
                        continue;
                    }
                    data[h] = d;
                    flag[h] = true;
                    ++s;
                    return itr(h, this);
                }
                if (dtok(data[h]) == dtok(d)) {
                    if (dflag[h] == true) {
                        data[h] = d;
                        dflag[h] = false;
                        ++s;
                    }
                    return itr(h, this);
                }
                h = (h + 1) & (cap - 1);
            }
        }

        // tips for speed up :
        // if return value is unnecessary, make argument_2 false.
        itr erase(itr it, bool get_next = true) {
            if (it == this->end()) return this->end();
            s--;
            if (shrink_rate(s)) {
                Data d = data[it.i];
                shrink();
                it = find(dtok(d));
            }
            int ni = (it.i + 1) & (cap - 1);
            if (this->flag[ni]) {
                this->dflag[it.i] = true;
            }
            else {
                this->flag[it.i] = false;
            }
            if (get_next) ++it;
            return it;
        }

        itr erase(const Key& key) { return erase(find(key)); }

        bool empty() const { return s == 0; }

        int size() const { return s; }

        void clear() {
            fill(std::begin(flag), std::end(flag), false);
            fill(std::begin(dflag), std::end(dflag), false);
            s = 0;
        }

        void reserve(int n) {
            if (n <= 0) return;
            n = 1 << min(23, __lg(n) + 2);
            if (cap < u32(n)) reallocate(n);
        }
    };

    template <typename Key, typename Data>
    uint64_t HashMapBase<Key, Data>::r =
        chrono::duration_cast<chrono::nanoseconds>(
            chrono::high_resolution_clock::now().time_since_epoch())
        .count();

}  // namespace HashMapImpl

template <typename Key>
struct HashSet : HashMapImpl::HashMapBase<Key, Key> {
    using HashMapImpl::HashMapBase<Key, Key>::HashMapBase;
};

/** compro_io **/

/* tuple */
// out
namespace aux {
    template<typename T, unsigned N, unsigned L>
    struct tp {
        static void output(std::ostream& os, const T& v) {
            os << std::get<N>(v) << ", ";
            tp<T, N + 1, L>::output(os, v);
        }
    };
    template<typename T, unsigned N>
    struct tp<T, N, N> {
        static void output(std::ostream& os, const T& v) { os << std::get<N>(v); }
    };
}
template<typename... Ts>
std::ostream& operator<<(std::ostream& os, const std::tuple<Ts...>& t) {
    os << '[';
    aux::tp<std::tuple<Ts...>, 0, sizeof...(Ts) - 1>::output(os, t);
    return os << ']';
}

template<class Ch, class Tr, class Container>
std::basic_ostream<Ch, Tr>& operator<<(std::basic_ostream<Ch, Tr>& os, const Container& x);

/* pair */
// out
template<class S, class T>
std::ostream& operator<<(std::ostream& os, const std::pair<S, T>& p) {
    return os << "[" << p.first << ", " << p.second << "]";
}
// in
template<class S, class T>
std::istream& operator>>(std::istream& is, std::pair<S, T>& p) {
    return is >> p.first >> p.second;
}

/* container */
// out
template<class Ch, class Tr, class Container>
std::basic_ostream<Ch, Tr>& operator<<(std::basic_ostream<Ch, Tr>& os, const Container& x) {
    bool f = true;
    os << "[";
    for (auto& y : x) {
        os << (f ? "" : ", ") << y;
        f = false;
    }
    return os << "]";
}
// in
template <
    class T,
    class = decltype(std::begin(std::declval<T&>())),
    class = typename std::enable_if<!std::is_same<T, std::string>::value>::type
>
std::istream& operator>>(std::istream& is, T& a) {
    for (auto& x : a) is >> x;
    return is;
}

std::ostream& operator<<(std::ostream& os, const std::vector<bool>& v) {
    std::string s(v.size(), ' ');
    for (int i = 0; i < v.size(); i++) s[i] = v[i] + '0';
    os << s;
    return os;
}

/* struct */
template<typename T>
auto operator<<(std::ostream& out, const T& t) -> decltype(out << t.stringify()) {
    out << t.stringify();
    return out;
}

/* setup */
struct IOSetup {
    IOSetup(bool f) {
        if (f) { std::cin.tie(nullptr); std::ios::sync_with_stdio(false); }
        std::cout << std::fixed << std::setprecision(15);
    }
} iosetup(true);

/** string formatter **/
template<typename... Ts>
std::string format(const std::string& f, Ts... t) {
    size_t l = std::snprintf(nullptr, 0, f.c_str(), t...);
    std::vector<char> b(l + 1);
    std::snprintf(&b[0], l + 1, f.c_str(), t...);
    return std::string(&b[0], &b[0] + l);
}

template<typename T>
std::string stringify(const T& x) {
    std::ostringstream oss;
    oss << x;
    return oss.str();
}

/* dump */
#define DUMPOUT std::cerr
std::ostringstream DUMPBUF;
#define dump(...) do{DUMPBUF<<"  ";DUMPBUF<<#__VA_ARGS__<<" :[DUMP - "<<__LINE__<<":"<<__FUNCTION__<<"]"<<std::endl;DUMPBUF<<"    ";dump_func(__VA_ARGS__);DUMPOUT<<DUMPBUF.str();DUMPBUF.str("");DUMPBUF.clear();}while(0);
void dump_func() { DUMPBUF << std::endl; }
template <class Head, class... Tail> void dump_func(Head&& head, Tail&&... tail) { DUMPBUF << head; if (sizeof...(Tail) == 0) { DUMPBUF << " "; } else { DUMPBUF << ", "; } dump_func(std::move(tail)...); }

/* timer */
class Timer {
    double t = 0, paused = 0, tmp;
public:
    Timer() { reset(); }
    static double time() {
#ifdef _MSC_VER
        return __rdtsc() / 3.0e9;
#else
        unsigned long long a, d;
        __asm__ volatile("rdtsc"
            : "=a"(a), "=d"(d));
        return (d << 32 | a) / 3.0e9;
#endif
    }
    void reset() { t = time(); }
    void pause() { tmp = time(); }
    void restart() { paused += time() - tmp; }
    double elapsed_ms() const { return (time() - t - paused) * 1000.0; }
};

/* rand */
struct Xorshift {
    uint64_t x = 88172645463325252LL;
    void set_seed(unsigned seed, int rep = 100) { x = uint64_t((seed + 1) * 10007); for (int i = 0; i < rep; i++) next_int(); }
    unsigned next_int() { x = x ^ (x << 7); return x = x ^ (x >> 9); }
    unsigned next_int(unsigned mod) { x = x ^ (x << 7); x = x ^ (x >> 9); return x % mod; }
    unsigned next_int(unsigned l, unsigned r) { x = x ^ (x << 7); x = x ^ (x >> 9); return x % (r - l + 1) + l; } // inclusive
    double next_double() { return double(next_int()) / UINT_MAX; }
} rnd;

/* shuffle */
template<typename T>
void shuffle_vector(std::vector<T>& v, Xorshift& rnd) {
    int n = v.size();
    for (int i = n - 1; i >= 1; i--) {
        int r = rnd.next_int(i);
        std::swap(v[i], v[r]);
    }
}

/* split */
std::vector<std::string> split(std::string str, const std::string& delim) {
    for (char& c : str) if (delim.find(c) != std::string::npos) c = ' ';
    std::istringstream iss(str);
    std::vector<std::string> parsed;
    std::string buf;
    while (iss >> buf) parsed.push_back(buf);
    return parsed;
}

template<typename A, size_t N, typename T> inline void Fill(A(&array)[N], const T& val) {
    std::fill((T*)array, (T*)(array + N), val);
}

template<typename T, typename ...Args> auto make_vector(T x, int arg, Args ...args) { if constexpr (sizeof...(args) == 0)return std::vector<T>(arg, x); else return std::vector(arg, make_vector<T>(x, args...)); }
template<typename T> bool chmax(T& a, const T& b) { if (a < b) { a = b; return true; } return false; }
template<typename T> bool chmin(T& a, const T& b) { if (a > b) { a = b; return true; } return false; }

class FastQueue {
    int front, back;
    int v[1 << 12];
public:
    FastQueue() : front(0), back(0) {}
    inline bool empty() { return front == back; }
    inline void push(int x) { v[front++] = x; }
    inline int pop() { return v[back++]; }
    inline void reset() { front = back = 0; }
    inline int size() { return front - back; }
} fqu;

using ll = long long;
using ld = double;
//using ld = boost::multiprecision::cpp_bin_float_quad;
using pii = std::pair<int, int>;
using pll = std::pair<ll, ll>;

using std::cin, std::cout, std::cerr, std::endl, std::string, std::vector;

constexpr char d2c[] = { 'L','U','R','D' };
int c2d[256];
constexpr char t2c[] = { '0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f' };
int c2t[256];
constexpr int LEFT = 1, UP = 2, RIGHT = 4, DOWN = 8;
constexpr int di[] = { 0, -1, 0, 1 };
constexpr int dj[] = { -1, 0, 1, 0 };

constexpr int8_t d4[] = { -1, -16, 1, 16 }; // LURD
inline uint8_t pack_p(uint8_t i, uint8_t j) { return ((i + 1) << 4) | (j + 1); }
inline uint16_t pack_e(uint8_t p1, uint8_t p2) { return (uint16_t(p1) << 8) | p2; }
inline uint16_t pack_e(uint8_t i1, uint8_t j1, uint8_t i2, uint8_t j2) { return pack_e(pack_p(i1, j1), pack_p(i2, j2)); }
inline uint8_t extract_i(uint8_t p) { return (p >> 4) - 1; }
inline uint8_t extract_j(uint8_t p) { return (p & 0xF) - 1; }
inline std::pair<uint8_t, uint8_t> unpack_p(uint8_t p) { return { extract_i(p), extract_j(p) }; }
inline std::pair<uint8_t, uint8_t> unpack_e2p(uint16_t e) { return { e >> 8, e & 0xFF }; }
inline std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> unpack_e2ij(uint16_t ij2) { return { ij2 >> 12, ij2 >> 8 & 0xF, ij2 >> 4 & 0xF, ij2 & 0xF }; }

struct UnionFind {
    vector<int> data;
    UnionFind() = default;
    explicit UnionFind(size_t sz) : data(sz, -1) {}
    bool unite(int x, int y) {
        x = find(x), y = find(y);
        if (x == y) return false;
        if (data[x] > data[y]) std::swap(x, y);
        data[x] += data[y];
        data[y] = x;
        return true;
    }
    int find(int k) {
        if (data[k] < 0) return (k);
        return data[k] = find(data[k]);
    }
    int size(int k) { return -data[find(k)]; }
    bool same(int x, int y) { return find(x) == find(y); }
};

struct RawInput {
    int N, T;
    vector<vector<int>> tiles;
};

struct Input {

    uint16_t N, T;
    uint8_t tiles[192];

    Input() {}

    Input(std::istream& in) {
        std::fill(tiles, tiles + 192, UCHAR_MAX);
        in >> N >> T;
        vector<string> S(N);
        in >> S;
        for (uint8_t i = 0; i < N; i++) {
            for (uint8_t j = 0; j < N; j++) {
                tiles[pack_p(i, j)] = c2t[S[i][j]];
            }
        }
    }

    // TODO: 空マスが右下以外の木の生成
    Input(uint16_t N_, Xorshift& rnd, uint8_t ei, uint8_t ej) : N(N_), T(2 * N * N * N) {
        std::fill(tiles, tiles + 192, UCHAR_MAX);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                tiles[pack_p(i, j)] = 0;
            }
        }
        vector<uint16_t> edges;
        for (uint8_t i = 0; i < N; i++) {
            for (uint8_t j = 0; j < N; j++) {
                if (i + 1 < N && !((i == ei || i + 1 == ei) && j == ej)) {
                    edges.push_back(pack_e(i, j, i + 1, j));
                }
                if (j + 1 < N && !(i == ei && (j == ej || j + 1 == ej))) {
                    edges.push_back(pack_e(i, j, i, j + 1));
                }
            }
        }
        shuffle_vector(edges, rnd);
        UnionFind uf(192);
        for (const uint16_t edge : edges) {
            auto [p1, p2] = unpack_e2p(edge);
            if (!uf.same(p1, p2)) {
                uf.unite(p1, p2);
                if (extract_i(p1) == extract_i(p2)) { // same row
                    tiles[p1] |= RIGHT;
                    tiles[p2] |= LEFT;
                }
                else { // same col
                    tiles[p1] |= DOWN;
                    tiles[p2] |= UP;
                }
            }
        }
    }

    string stringify() const {
        string res;
        res += "N = " + std::to_string(N) + '\n';
        res += "T = " + std::to_string(N) + '\n';
        for (uint8_t i = 0; i < N; i++) {
            for (uint8_t j = 0; j < N; j++) {
                res += format("%3d ", (int)tiles[pack_p(i, j)]);
            }
            res += '\n';
        }
        return res;
    }

    RawInput cvt() const {
        RawInput in;
        in.N = N;
        in.T = T;
        in.tiles.resize(N, vector<int>(N));
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                in.tiles[i][j] = tiles[pack_p(i, j)];
            }
        }
        return in;
    }

};

namespace NJudge {
    struct Sim {
        uint64_t n, T, turn, i, j;
        vector<vector<std::pair<uint64_t, uint64_t>>> from;
        Sim(const RawInput& input) : n(input.N), T(input.T), from(input.N, vector<std::pair<uint64_t, uint64_t>>(input.N, std::make_pair(0ULL, 0ULL))) {
            i = -1; j = -1;
            for (uint64_t x = 0; x < n; x++) {
                for (uint64_t y = 0; y < n; y++) {
                    if (input.tiles[x][y] == 0) {
                        i = x; j = y;
                    }
                    from[x][y] = { x, y };
                }
            }
            turn = 0;
        }
        bool apply(char c) {
            int d = c2d[c];
            auto i2 = i + di[d];
            auto j2 = j + dj[d];
            if (i2 >= n || j2 >= n) {
                //cerr << format("illegal move: %c (turn %lld)\n", c, turn);
                return false;
            }
            auto f1 = from[i][j];
            auto f2 = from[i2][j2];
            from[i2][j2] = f1;
            from[i][j] = f2;
            i = i2;
            j = j2;
            turn++;
            return true;
        }
        int compute_score(const RawInput& input) {
            UnionFind uf(n * n);
            vector<bool> tree(n * n, true);
            vector<vector<int>> tiles(n, vector<int>(n, 0));
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    tiles[i][j] = input.tiles[from[i][j].first][from[i][j].second];
                }
            }
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (i + 1 < n && (tiles[i][j] & 8) != 0 && (tiles[i + 1][j] & 2) != 0) {
                        int a = uf.find(i * n + j);
                        int b = uf.find((i + 1) * n + j);
                        if (a == b) {
                            tree[a] = false;
                        }
                        else {
                            int t = tree[a] && tree[b];
                            uf.unite(a, b);
                            tree[uf.find(a)] = t;
                        }
                    }
                    if (j + 1 < n && (tiles[i][j] & 4) != 0 && (tiles[i][j + 1] & 1) != 0) {
                        int a = uf.find(i * n + j);
                        int b = uf.find(i * n + (j + 1));
                        if (a == b) {
                            tree[a] = false;
                        }
                        else {
                            int t = tree[a] && tree[b];
                            uf.unite(a, b);
                            tree[uf.find(a)] = t;
                        }
                    }
                }
            }
            uint64_t max_tree = -1;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (tiles[i][j] != 0 && tree[uf.find(i * n + j)]) {
                        if (max_tree == -1 || uf.size(max_tree) < uf.size(i * n + j)) {
                            max_tree = i * n + j;
                        }
                    }
                }
            }
            if (turn > T) {
                //cerr << format("too many moves\n");
                return 0;
            }
            if (max_tree == -1) {
                return 0;
            }
            auto size = uf.size(max_tree);
            if (size == n * n - 1) {
                return (int)round(500000.0 * (1.0 + double(T - turn) / T));
            }
            return (int)round(500000.0 * size / (n * n - 1));
        }
    };
    int compute_score(const RawInput& input, const string& out) {
        Sim sim(input);
        for (char c : out) {
            if (!sim.apply(c)) {
                return 0;
            }
        }
        return sim.compute_score(input);
    }
}

struct TreeModifier {

    uint16_t N;
    uint8_t target_ctr[16];
    uint8_t tree_ctr[16];
    uint8_t tiles[192];
    vector<uint16_t> disabled;
    uint16_t cost;

    bool seen[192];
    uint8_t prev[192];

    TreeModifier(const Input& input, const Input& tree) : N(input.N) {
        std::fill(target_ctr, target_ctr + 16, 0);
        std::fill(tree_ctr, tree_ctr + 16, 0);
        std::memcpy(tiles, tree.tiles, sizeof(uint8_t) * 192);
        for (uint8_t i = 0; i < N; i++) {
            for (uint8_t j = 0; j < N; j++) {
                target_ctr[input.tiles[pack_p(i, j)]]++;
                tree_ctr[tiles[pack_p(i, j)]]++;
            }
        }
        for (uint8_t i = 0; i < N; i++) {
            for (uint8_t j = 0; j < N - 1; j++) {
                auto t1 = tiles[pack_p(i, j)], t2 = tiles[pack_p(i, j + 1)];
                if (!((t1 & RIGHT) * (t2 & LEFT)) && !(i == N - 1 && j == N - 2)) {
                    disabled.push_back(pack_e(i, j, i, j + 1));
                }
            }
        }
        for (uint8_t i = 0; i < N - 1; i++) {
            for (uint8_t j = 0; j < N; j++) {
                auto t1 = tiles[pack_p(i, j)], t2 = tiles[pack_p(i + 1, j)];
                if (!((t1 & DOWN) * (t2 & UP)) && !(i == N - 2 && j == N - 1)) {
                    disabled.push_back(pack_e(i, j, i + 1, j));
                }
            }
        }
        cost = 0;
        for (uint8_t t = 0; t < 16; t++) {
            cost += abs((int)target_ctr[t] - (int)tree_ctr[t]);
        }
    }

    uint16_t choose_random_disabled_edge(Xorshift& rnd) {
        int eid = rnd.next_int(disabled.size());
        std::swap(disabled[eid], disabled.back()); // 末端に移動しておく
        auto e = disabled.back();
        return e;
    }

    void toggle_edge(uint8_t p1, uint8_t p2) {
        tree_ctr[tiles[p1]]--;
        tree_ctr[tiles[p2]]--;
        if (extract_i(p1) == extract_i(p2)) {
            tiles[p1] ^= RIGHT;
            tiles[p2] ^= LEFT;
        }
        else {
            tiles[p1] ^= DOWN;
            tiles[p2] ^= UP;
        }
        tree_ctr[tiles[p1]]++;
        tree_ctr[tiles[p2]]++;
    }

    void local_search(Xorshift& rnd) {
        // 接続していない辺を on にする
        // 出来たサイクルの辺を一つ選択して off にする
        auto e = choose_random_disabled_edge(rnd);
        uint8_t p1, p2;
        std::tie(p1, p2) = unpack_e2p(e);
        // (i1,j1) -> (i2,j2) のパスを求める
        auto cands = [&]() {
            std::fill(seen, seen + 192, false);
            std::fill(prev, prev + 192, UCHAR_MAX);
            fqu.reset();
            fqu.push(p1);
            seen[p1] = true;
            bool found = false;
            while (!fqu.empty()) {
                uint8_t p = fqu.pop(), i = extract_i(p), j = extract_j(p);
                auto t = tiles[p];
                for (int d = 0; d < 4; d++) {
                    if (!(t >> d & 1)) continue; // d 方向に伸びていない
                    uint8_t np = p + d4[d];
                    if (seen[np]) continue;
                    seen[np] = true;
                    fqu.push(np);
                    prev[np] = p;
                    if (p2 == np) {
                        found = true;
                        break;
                    }
                }
                if (found) break;
            }
            auto p = p2;
            vector<uint16_t> edges; edges.reserve(128);
            while (true) {
                auto np = prev[p];
                if (np == UCHAR_MAX) break;
                edges.push_back(pack_e(std::min(p, np), std::max(p, np)));
                p = np;
            }
            return edges;
        }();

        uint16_t min_cost = std::numeric_limits<uint16_t>::max();
        uint16_t min_edge;
        // connect (i1,j1)-(i2,j2)
        toggle_edge(p1, p2);
        for (auto e2 : cands) {
            auto [p3, p4] = unpack_e2p(e2);
            // disconnect (i3,j3)-(i4,j4)
            toggle_edge(p3, p4);
            uint16_t new_cost = 0;
            for (int t = 0; t < 16; t++) {
                new_cost += abs((int)target_ctr[t] - (int)tree_ctr[t]);
            }

            auto diff = (int)new_cost - (int)cost;
            double temp = 0.3;
            double prob = exp(-diff / temp);

            if (rnd.next_double() < prob) {
                disabled.pop_back();
                disabled.emplace_back(pack_e(p3, p4));
                cost = new_cost;
                return;
            }

            if (chmin(min_cost, new_cost)) {
                min_edge = e2;
            }
            toggle_edge(p3, p4);
        }

        toggle_edge(p1, p2); // revert
    }

    RawInput cvt() const {
        RawInput in;
        in.N = N;
        in.T = (int)2 * N * N * N;
        in.tiles.resize(N, vector<int>(N));
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                in.tiles[i][j] = tiles[pack_p(i, j)];
            }
        }
        return in;
    }

};


// linear conflict
struct LCSolver {
    int N, e, ctr;
    int p[10];
    int nadj[10];
    bool adj[10][10];

    void setup(int N) {
        this->N = N;
        e = pack_p(N - 1, N - 1);
    }

    int calc() {
        std::memset(nadj, 0, sizeof(int) * ctr);
        std::memset(adj, 0, sizeof(bool) * 10 * ctr);
        for (int i = 0; i + 1 < ctr; i++) {
            for (int j = i + 1; j < ctr; j++) {
                bool b = p[i] > p[j];
                adj[i][j] = adj[j][i] = b;
                nadj[i] += b;
                nadj[j] += b;
            }
        }
        int lc = 0;
        while (true) {
            auto it = std::max_element(nadj, nadj + ctr);
            if (*it == 0) break;
            int u = std::distance(nadj, it);
            for (int v = 0; v < ctr; v++) {
                if (adj[u][v]) {
                    adj[u][v] = adj[v][u] = false;
                    nadj[u]--;
                    nadj[v]--;
                }
            }
            lc++;
        }
        return lc << 1;
    }

    int calc_row(const uint8_t* tiles, int row) {
        ctr = 0;
        int begin = pack_p(row, 0), end = begin + N;
        for (int pos = begin; pos < end; ++pos) {
            int t = tiles[pos];
            if (t != e && extract_i(t) == row) {
                p[ctr++] = t;
            }
        }
        if (ctr <= 1) return 0;
        return calc();
    }

    int calc_col(const uint8_t* tiles, int col) {
        ctr = 0;
        int begin = pack_p(0, col), end = begin + (N << 4);
        for (int pos = begin; pos < end; pos += 16) {
            int t = tiles[pos];
            if (t != e && extract_j(t) == col) {
                p[ctr++] = t;
            }
        }
        if (ctr <= 1) return 0;
        return calc();
    }

    int calc_all(const uint8_t* tiles) {
        int res = 0;
        for (int row = 0; row < N; row++) res += calc_row(tiles, row);
        for (int col = 0; col < N; col++) res += calc_col(tiles, col);
        return res;
    }

} g_lcsol;

namespace NFlow {

    template< typename T >
    std::pair<T, vector<int>> hungarian(vector<vector<T>>& A) {
        const T infty = std::numeric_limits<T>::max();
        const int N = (int)A.size();
        const int M = (int)A[0].size();
        vector<int> P(M), way(M);
        vector<T> U(N, 0), V(M, 0), minV;
        vector<bool> used;

        for (int i = 1; i < N; i++) {
            P[0] = i;
            minV.assign(M, infty);
            used.assign(M, false);
            int j0 = 0;
            while (P[j0] != 0) {
                int i0 = P[j0], j1 = 0;
                used[j0] = true;
                T delta = infty;
                for (int j = 1; j < M; j++) {
                    if (used[j]) continue;
                    T curr = A[i0][j] - U[i0] - V[j];
                    if (curr < minV[j]) minV[j] = curr, way[j] = j0;
                    if (minV[j] < delta) delta = minV[j], j1 = j;
                }
                for (int j = 0; j < M; j++) {
                    if (used[j]) U[P[j]] += delta, V[j] -= delta;
                    else minV[j] -= delta;
                }
                j0 = j1;
            }
            do {
                P[j0] = P[way[j0]];
                j0 = way[j0];
            } while (j0 != 0);
        }
        return { -V[0], P };
    }

    struct Result {
        int total_cost;
        uint8_t tiles[192];
    };

    Result calc_assign(const Input& input, const TreeModifier& tmod) {
        vector<vector<pii>> from(16, { {-1,-1} }), to(16, { {-1,-1} });
        int N = input.N;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                from[input.tiles[pack_p(i, j)]].emplace_back(i, j);
                to[tmod.tiles[pack_p(i, j)]].emplace_back(i, j);
            }
        }
        Result res;
        res.total_cost = 0;
        for (int t = 0; t < 16; t++) {
            const auto& fps = from[t];
            const auto& tps = to[t];
            if (fps.size() == 1) continue;
            if (fps.size() == 2) {
                auto [fi, fj] = fps[1];
                auto [ti, tj] = tps[1];
                res.total_cost += abs(fi - ti) + abs(fj - tj);
                res.tiles[pack_p(fi, fj)] = pack_p(ti, tj);
                continue;
            }
            auto dist = make_vector(0, fps.size(), tps.size());
            for (int i = 1; i < fps.size(); i++) {
                auto [fi, fj] = fps[i];
                for (int j = 1; j < tps.size(); j++) {
                    auto [ti, tj] = tps[j];
                    dist[i][j] = abs(fi - ti) + abs(fj - tj);
                }
            }
            auto [cost, p] = hungarian(dist);
            res.total_cost += cost;
            for (int i = 1; i < fps.size(); i++) {
                res.tiles[pack_p(fps[p[i]].first, fps[p[i]].second)] = pack_p(tps[i].first, tps[i].second);
        }
        }
        g_lcsol.setup(N);
        res.total_cost += g_lcsol.calc_all(res.tiles);
        return res;
    }

}


namespace NBeam {

    constexpr int max_beam_turn = 512;

    // 1-indexed にして out of bound 判定を速くする
    // 省メモリ
    // 12 x 16

    uint64_t g_hash_table[192][192];

    struct HashSetup {
        HashSetup() {
            std::mt19937_64 engine;
            for (int p = 0; p < 192; p++) {
                for (int n = 0; n < 192; n++) {
                    g_hash_table[p][n] = engine();
                }
            }
        }
    } hash_setup;

    struct State;
    using StatePtr = std::shared_ptr<State>;
    struct State {

        int16_t turn;
        uint8_t tiles[192];
        int16_t md, lc;
        uint8_t lc_r[10], lc_c[10];
        uint8_t ep;
        int8_t pdir;
        uint8_t cmds[max_beam_turn >> 2]; // 1 ターン 2bit
        uint64_t hash;

        State() : md(SHRT_MAX) {}

        State(int N, int T, const vector<std::tuple<int, int, int, int>>& assign) {

            turn = md = lc = 0;
            std::fill(tiles, tiles + 192, UCHAR_MAX);
            ep = -1;
            pdir = -1;
            std::fill(cmds, cmds + (max_beam_turn >> 2), 0);
            hash = 0;

            g_lcsol.setup(N);

            int tp_empty = pack_p(N - 1, N - 1);
            for (auto [si, sj, ti, tj] : assign) {
                int sp = pack_p(si, sj);
                int tp = pack_p(ti, tj);
                tiles[sp] = tp;
                if (tp == tp_empty) {
                    ep = sp;
                    continue;
                }
                md += cell_cost(sp);
            }
            for (int r = 0; r < N; r++) {
                lc_r[r] = g_lcsol.calc_row(tiles, r);
                lc += lc_r[r];
            }
            for (int c = 0; c < N; c++) {
                lc_c[c] = g_lcsol.calc_col(tiles, c);
                lc += lc_c[c];
            }

        }

        inline int cost() const {
            return md + lc;
        }

        void print() const {
            cerr << format("turn=%d, md=%d, lc=%d, ep=%d, pdir=%d, hash=%lld\n", turn, md, lc, ep, pdir, hash);
        }

        inline int cell_cost(int p) const {
            int tp = tiles[p];
            return abs(extract_i(p) - extract_i(tp)) + abs(extract_j(p) - extract_j(tp));
        }

        inline bool can_move(int dir) const {
            return ((dir + 2) & 3) != pdir && tiles[ep + d4[dir]] != UCHAR_MAX;
        }

        inline uint64_t move_hash(int dir) const {
            int np = ep + d4[dir];
            return hash ^ g_hash_table[np][tiles[np]] ^ g_hash_table[ep][tiles[np]];
            }

        inline void move(int dir) {
            int np = ep + d4[dir];
            if (dir & 1) lc -= lc_r[extract_i(np)] + lc_r[extract_i(ep)];
            else lc -= lc_c[extract_j(np)] + lc_c[extract_j(ep)];
            md -= cell_cost(np);
            hash ^= g_hash_table[np][tiles[np]];
            std::swap(tiles[ep], tiles[np]);
            hash ^= g_hash_table[ep][tiles[ep]];
            md += cell_cost(ep);
            if (dir & 1) {
                lc_r[extract_i(np)] = g_lcsol.calc_row(tiles, extract_i(np));
                lc_r[extract_i(ep)] = g_lcsol.calc_row(tiles, extract_i(ep));
                lc += lc_r[extract_i(np)] + lc_r[extract_i(ep)];
            }
            else {
                lc_c[extract_j(np)] = g_lcsol.calc_col(tiles, extract_j(np));
                lc_c[extract_j(ep)] = g_lcsol.calc_col(tiles, extract_j(ep));
                lc += lc_c[extract_j(np)] + lc_c[extract_j(ep)];
            }
            ep = np;
            pdir = dir;
            // 1 byte に 4 個
            cmds[turn >> 2] ^= dir << ((turn & 3) << 1);
            turn++;
        }

        string get_cmd() const {
            string res;
            for (int t = 0; t < turn; t++) {
                res += d2c[(cmds[t >> 2] >> ((t & 3) << 1)) & 3];
            }
            return res;
        }

    };

    bool equals(const State& s1, const State& s2) {
        if (s1.turn != s2.turn) return false;
        if (s1.md != s2.md) return false;
        if (s1.ep != s2.ep) return false;
        if (s1.pdir != s2.pdir) return false;
        if (s1.hash != s2.hash) return false;
        if (memcmp(s1.tiles, s2.tiles, sizeof(uint8_t) * 192)) return false;
        if (memcmp(s1.cmds, s2.cmds, sizeof(uint8_t) * (max_beam_turn >> 2))) return false;
        return true;
    }

    State beam_search(State init_state, int beam_width, double duration) {
        static constexpr int max_beam_width = 15000, degree = 4;
        static State sbuf[2][max_beam_width * degree];
        static int ord[max_beam_width * degree];

        Timer timer;

        int now_buffer = 0;
        int buf_size[2] = {};

        sbuf[now_buffer][0] = init_state;
        ord[0] = 0;
        buf_size[now_buffer]++;

        State best_state(init_state);
        HashSet<unsigned long long> seen; seen.reserve(30000000);
        seen.insert(init_state.hash);

        int turn = 0;
        while (buf_size[now_buffer] && turn < max_beam_turn && timer.elapsed_ms() < duration && best_state.md) {

            auto& now_states = sbuf[now_buffer];
            auto& now_size = sbuf[now_buffer];
            auto& next_states = sbuf[now_buffer ^ 1];
            auto& next_size = buf_size[now_buffer ^ 1]; next_size = 0;

            for (int n = 0; n < std::min(beam_width, buf_size[now_buffer]); n++) {
                auto& now_state = now_states[ord[n]];
                for (int d = 0; d < 4; d++) {
                    if (!now_state.can_move(d)) continue;
                    auto& next_state = next_states[next_size];
                    next_state = now_state;
                    next_state.move(d);
                    if (!seen.contain(next_state.hash)) {
                        seen.insert(next_state.hash);
                        next_size++;
                    }
                }
            }

            if (!next_size) break;
            std::iota(ord, ord + next_size, 0);
            std::sort(ord, ord + next_size, [&next_states](int a, int b) {
                return next_states[a].cost() < next_states[b].cost();
                });

            if (next_states[ord[0]].md < best_state.md) {
                best_state = next_states[ord[0]];
                //dump(turn, best_state.md);
            }

            now_buffer ^= 1; // toggle buffer
            turn++;
        }
        dump(seen.size());
        return best_state;
    }

}

namespace NPuzzle {

    constexpr int di8[] = { 0, -1, -1, -1, 0, 1, 1, 1 };
    constexpr int dj8[] = { -1, -1, 0, 1, 1, 1, 0, -1 };

    struct State {

        Xorshift rnd;

        int N; // board size
        int ei, ej; // empty cell position
        vector<vector<int>> tiles; // numbers on cell
        vector<vector<bool>> fixed;
        int md; // manhattan distance
        string cmds; // length = num turns

        State() {}

        State(int N, const NFlow::Result& assign) : N(N) {
            tiles.resize(N, vector<int>(N, N * N - 1));
            fixed.resize(N, vector<bool>(N));
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    int t = assign.tiles[pack_p(i, j)], ti = extract_i(t), tj = extract_j(t);
                    tiles[i][j] = ti * N + tj;
                }
            }
            md = calc_md_naive();
            std::tie(ei, ej) = get_pos(N * N - 1);
        }

        void output_problem(std::ostream& out) const {
            out << N << '\n';
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    out << tiles[i][j] << ' ';
                }
                out << '\n';
            }
        }

#ifdef HAVE_OPENCV_HIGHGUI
        void show(int delay = 0) {
            int N = tiles.size();
            int sz = 800 / N, H = sz * N, W = sz * N;
            cv::Mat_<cv::Vec3b> img(H, W, cv::Vec3b(255, 255, 255));
            int max_md = (N - 1) * 2;
            cv::Scalar red(0, 0, 255), white(255, 255, 255);
            auto get_color = [&](int md) {
                return red * md / double(max_md) + white * (max_md - md) / double(max_md);
            };
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    if (tiles[i][j] == N * N - 1) {
                        cv::rectangle(img, cv::Rect(j * sz, i * sz, sz, sz), cv::Scalar(200, 200, 200), cv::FILLED);
                    }
                    else {
                        int t = tiles[i][j], ti = t / N, tj = t % N, md = abs(i - ti) + abs(j - tj);
                        cv::rectangle(img, cv::Rect(j * sz, i * sz, sz, sz), get_color(md), cv::FILLED);
                        cv::Point ctr(j * sz + sz / 3, i * sz + sz / 2);
                        cv::putText(img, std::to_string(tiles[i][j]), ctr, cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);
                    }
                }
            }
            for (int i = 0; i <= N; i++) {
                cv::line(img, cv::Point(0, i * sz), cv::Point(W, i * sz), cv::Scalar(200, 200, 200), 2);
                cv::line(img, cv::Point(i * sz, 0), cv::Point(i * sz, H), cv::Scalar(200, 200, 200), 2);
            }
            cv::imshow("img", img);
            cv::waitKey(delay);
        }

        void animate(int delay = 0) {
            //dump(cmds);
            auto tmp_cmds = cmds;
            while (cmds.size()) undo();
            show();
            for (char c : tmp_cmds) {
                move(c2d[c]);
                show(delay);
            }
            show();
        }
#endif

        inline pii get_pos(int n) const {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    if (tiles[i][j] == n) {
                        return { i, j };
                    }
                }
            }
            assert(false);
            return { -1, -1 };
        }

        void move_empty_cell(int ti, int tj) {
            if (ei == ti && ej == tj) {
                return;
            }
            // empty cell を (ti,tj) まで動かす
            bool seen[10][10];
            pii prev[10][10];
            Fill(seen, false);
            Fill(prev, pii(-1, -1));
            std::queue<pii> qu({ {ei, ej} }); // (i,j,prev_dir)
            seen[ei][ej] = true;
            while (!qu.empty()) {
                auto [i, j] = qu.front(); qu.pop();
                if (i == ti && j == tj) break;
                for (int d = 0; d < 4; d++) {
                    int ni = i + di[d], nj = j + dj[d];
                    if (ni < 0 || ni >= N || nj < 0 || nj >= N || fixed[ni][nj] || seen[ni][nj]) continue;
                    seen[ni][nj] = true;
                    prev[ni][nj] = { i, j };
                    qu.emplace(ni, nj);
                }
            }
            vector<pii> path;
            {
                pii p(ti, tj);
                path.push_back(p);
                while (p.first != -1) {
                    p = prev[p.first][p.second];
                    if (p.first == -1) break;
                    path.push_back(p);
                }
                reverse(path.begin(), path.end());
            }
            for (int i = 0; i + 1 < path.size(); i++) {
                auto [i1, j1] = path[i];
                auto [i2, j2] = path[i + 1];
                int dir = get_dir(i1, j1, i2, j2);
                move(dir);
            }
        }

        inline bool is_aligned(int n, int ti, int tj) const {
            auto [si, sj] = get_pos(n);
            return si == ti && sj == tj;
        }

        inline bool is_aligned(int n) const {
            return is_aligned(n, n / N, n % N);
        }

        void move_number(int n, int ti, int tj) {

            if (is_aligned(n, ti, tj)) return; // もう揃ってる

            // 数字 n を T(ti,tj) まで移動させる
            auto [si, sj] = get_pos(n); // 数字 n の現在位置 S

            // E -> S (の 4 近傍)のパスを求める
            // manhattan distance 増加するなら +2, そうでないなら 0
            // 01-BFS で
            constexpr int inf = INT_MAX / 4;
            vector<vector<int>> dist(N, vector<int>(N, inf));
            vector<vector<int>> prev(N, vector<int>(N, -1));
            std::deque<pii> dq({ {ei, ej} });
            dist[ei][ej] = 0;
            fixed[si][sj] = true; // S を fix
            while (!dq.empty()) {
                auto [i, j] = dq.front(); dq.pop_front();
                for (int d = 0; d < 4; d++) {
                    int ni = i + di[d], nj = j + dj[d];
                    if (ni < 0 || ni >= N || nj < 0 || nj >= N || fixed[ni][nj]) continue;
                    // (ni,nj) にあるタイルが (i,j) に移動した際のマンハッタン距離の変化 + 移動距離
                    int tile = tiles[ni][nj], dst_i = tile / N, dst_j = tile % N;
                    int cost = abs(dst_i - i) + abs(dst_j - j) - abs(dst_i - ni) - abs(dst_j - nj) + 1;
                    assert(cost == 0 || cost == 2);
                    if (chmin(dist[ni][nj], dist[i][j] + cost)) {
                        prev[ni][nj] = (d + 2) & 3;
                        if (cost) {
                            dq.emplace_back(ni, nj);
                        }
                        else {
                            dq.emplace_front(ni, nj);
                        }
                    }
                }
            }
            fixed[si][sj] = false;

            // S の 4 近傍のうち、T までのマンハッタン距離が最小となるような候補点を集める
            vector<int> cand_dirs;
            int min_dist = inf;
            {
                for (int d = 0; d < 4; d++) {
                    int i = si + di[d], j = sj + dj[d];
                    if (i < 0 || i >= N || j < 0 || j >= N || fixed[i][j]) continue;
                    int dist = abs(i - ti) + abs(j - tj);
                    if (dist < min_dist) {
                        cand_dirs.clear();
                        min_dist = dist;
                    }
                    if (dist == min_dist) {
                        cand_dirs.push_back(d);
                    }
                }
            }
            // min_dist はビームサーチのターン数になる

            // 空マスを候補点まで移動させた状態をビームサーチの初期状態にする
            vector<State> init_states;
            int checkpoint = cmds.size();
            for (int dd : cand_dirs) {
                vector<int> path;
                {
                    int i = si + di[dd], j = sj + dj[dd];
                    if (i < 0 || i >= N || j < 0 || j >= N || fixed[i][j]) continue;
                    int d = prev[i][j];
                    if (d != -1) {
                        path.push_back(d);
                        while (d != -1) {
                            i += di[d]; j += dj[d];
                            d = prev[i][j];
                            if (d == -1) break;
                            path.push_back(d);
                        }
                    }
                }
                for (int& d : path) d = (d + 2) & 3;
                std::reverse(path.begin(), path.end());
                for (int d : path) move(d);
                move((dd + 2) & 3); // S と交換するまでやってしまう
                init_states.push_back(*this);
                while (cmds.size() > checkpoint) undo();
            }

            // S -> T の最短経路をビームサーチ

            // S から見た空マスの方向(ed)と進行方向(td)の位置関係によって 3 通りの状況が発生する
            // S->ed と S->td が 90° の角をなしており、角の間が fix されていないとき
            //   3 手で S を移動させる方法が 1 通り
            // S->ed と S->td が 180°
            //   5 手で S を移動させる方法が 1 or 2 通り
            // S->ed と S->td が 90° の角をなしており、角の間が fix されているとき
            //   7 手で S を移動させる方法が 1 通り

            int width = 100;
            vector<State> now_states(init_states);
            for (int turn = 0; turn < min_dist; turn++) {
                vector<State> next_states;
                for (const auto& state : now_states) {
                    auto [si2, sj2] = state.get_pos(n);
                    int len = abs(si2 - ti) + abs(sj2 - tj);
                    for (int td = 0; td < 4; td++) {
                        int ti2 = si2 + di[td], tj2 = sj2 + dj[td];
                        int nlen = abs(ti2 - ti) + abs(tj2 - tj);
                        if (nlen == len - 1) {
                            for (const auto& next_state : state.next_state(si2, sj2, td)) {
                                next_states.push_back(next_state);
                            }
                        }
                    }
                }
                sort(next_states.begin(), next_states.end(), [](const State& a, const State& b) {
                    return a.cmds.size() * 2 + a.md < b.cmds.size() * 2 + b.md;
                    });
                while (next_states.size() > width) next_states.pop_back();
                now_states = next_states;
            }

            *this = now_states.front();
        }

        int get_dir(int si, int sj, int ti, int tj) const {
            if (si == ti) return sj < tj ? 2 : 0;
            return si < ti ? 3 : 1;
        }

        bool is_inside(int i, int j) const {
            return 0 <= i && i < N && 0 <= j && j < N;
        }

        vector<State> next_state(int si, int sj, int td) const {
            static constexpr int dcw[] = { 1,2,2,3,3,0,0,1 };
            static constexpr int dccw[] = { 3,3,0,0,1,1,2,2 };

            vector<State> res;

            int ed = get_dir(si, sj, ei, ej);
            assert(ed != td);

            // TODO: 短い方のみ採用
            bool cw_ok = true;
            for (int d = (ed * 2 + 1) & 7; d != ((td * 2 + 1) & 7); d = (d + 1) & 7) {
                if (!is_inside(si + di8[d], sj + dj8[d]) || fixed[si + di8[d]][sj + dj8[d]]) {
                    cw_ok = false;
                    break;
                }
            }
            if (cw_ok) {
                auto state(*this);
                for (int d = ed * 2; d != td * 2; d = (d + 1) & 7) {
                    state.move(dcw[d]);
                }
                int d = state.get_dir(state.ei, state.ej, si, sj);
                state.move(d);
                res.push_back(state);
            }
            bool ccw_ok = true;
            for (int d = (ed * 2 + 7) & 7; d != ((td * 2 + 7) & 7); d = (d + 7) & 7) {
                if (!is_inside(si + di8[d], sj + dj8[d]) || fixed[si + di8[d]][sj + dj8[d]]) {
                    ccw_ok = false;
                    break;
                }
            }
            if (ccw_ok) {
                auto state(*this);
                for (int d = ed * 2; d != td * 2; d = (d + 7) & 7) {
                    state.move(dccw[d]);
                }
                int d = state.get_dir(state.ei, state.ej, si, sj);
                state.move(d);
                res.push_back(state);
            }

            return res;
        }

        inline int IJ(int i, int j) const {
            return i * N + j;
        }

        void align_horizontal(int layer) {
            for (int j = layer + 1; j < N - 2; j++) {
                move_number(IJ(layer, j), layer, j);
                fixed[layer][j] = true;
            }

            if (is_aligned(IJ(layer, N - 2)) && is_aligned(IJ(layer, N - 1))) return; // 揃ってる

            // tile(layer,N-2) を (layer,N-1) に移動させたとき、tile(layer,N-1) が (layer,N-2) に来てしまう場合
            auto cpuz(*this);
            cpuz.move_number(IJ(layer, N - 2), layer, N - 1);
            if (cpuz.tiles[layer][N - 2] == IJ(layer, N - 1) || (cpuz.tiles[layer][N - 2] == IJ(N - 1, N - 1) && cpuz.tiles[layer + 1][N - 2] == IJ(layer, N - 1))) {
                move_number(IJ(layer, N - 2), layer, N - 2);
                fixed[layer][N - 2] = true;
                if (is_aligned(IJ(layer, N - 1))) {
                    fixed[layer][N - 1] = true;
                }
                else {
                    move_number(IJ(layer, N - 1), layer + 2, N - 1);
                    fixed[layer + 2][N - 1] = true;
                    move_empty_cell(layer + 1, N - 1);
                    fixed[layer + 2][N - 1] = false;
                    fixed[layer][N - 1] = true;
                    move("LURDLURDDLUURD");
                }
            }
            else {
                move_number(IJ(layer, N - 2), layer, N - 1);
                fixed[layer][N - 1] = true;
                move_number(IJ(layer, N - 1), layer + 1, N - 1);
                fixed[layer + 1][N - 1] = true;
                move_empty_cell(layer, N - 2);
                fixed[layer + 1][N - 1] = false;
                fixed[layer][N - 2] = true;
                move("RD");
            }
        }

        void align_vertical(int layer) {
            for (int i = layer + 1; i < N - 2; i++) {
                move_number(IJ(i, layer), i, layer);
                fixed[i][layer] = true;
            }

            if (is_aligned(IJ(N - 2, layer)) && is_aligned(IJ(N - 1, layer))) return; // 揃ってる

            auto cpuz(*this);
            cpuz.move_number(IJ(N - 2, layer), N - 1, layer);
            if (cpuz.tiles[N - 2][layer] == IJ(N - 1, layer) || (cpuz.tiles[N - 2][layer] == IJ(N - 1, N - 1) && cpuz.tiles[N - 2][layer + 1] == IJ(N - 1, layer))) {
                move_number(IJ(N - 2, layer), N - 2, layer);
                fixed[N - 2][layer] = true;
                if (is_aligned(IJ(N - 1, layer))) {
                    fixed[N - 1][layer] = true;
                }
                else {
                    move_number(IJ(N - 1, layer), N - 1, layer + 2);
                    fixed[N - 1][layer + 2] = true;
                    move_empty_cell(N - 1, layer + 1);
                    fixed[N - 1][layer + 2] = false;
                    fixed[N - 1][layer] = true;
                    move("ULDRULDRRULLDR");
                }
            }
            else {
                move_number(IJ(N - 2, layer), N - 1, layer);
                fixed[N - 1][layer] = true;
                move_number(IJ(N - 1, layer), N - 1, layer + 1);
                fixed[N - 1][layer + 1] = true;
                move_empty_cell(N - 2, layer);
                fixed[N - 1][layer + 1] = false;
                fixed[N - 2][layer] = true;
                move("DR");
            }
        }

        void align(int layer) {

            move_number(layer * N + layer, layer, layer);
            fixed[layer][layer] = true;

            int best_turn = INT_MAX;
            State best_state;

            // h->v vs. v->h
            {
                State s(*this);
                s.align_horizontal(layer);
                s.align_vertical(layer);
                if (chmin(best_turn, (int)s.cmds.size())) {
                    best_state = s;
                }
            }
            {
                State s(*this);
                s.align_vertical(layer);
                s.align_horizontal(layer);
                if (chmin(best_turn, (int)s.cmds.size())) {
                    best_state = s;
                }
            }

            *this = best_state;
        }

        std::pair<bool, int> can_move(int d) const {
            int ni = ei + di[d], nj = ej + dj[d];
            if (ni < 0 || ni >= N || nj < 0 || nj >= N) return { false, -1 };
            int t = tiles[ni][nj], ti = t / N, tj = t % N;
            return { true, abs(ei - ti) + abs(ej - tj) - abs(ni - ti) - abs(nj - tj) };
        }

        void move(int d) {
            int ni = ei + di[d], nj = ej + dj[d];
            int t = tiles[ni][nj], ti = t / N, tj = t % N;
            md += abs(ei - ti) + abs(ej - tj) - abs(ni - ti) - abs(nj - tj);
            std::swap(tiles[ei][ej], tiles[ni][nj]);
            ei = ni; ej = nj;
            cmds += d2c[d];
        }

        void move(const string& s) {
            for (char c : s) move(c2d[c]);
        }

        void undo() {
            int d = (c2d[cmds.back()] + 2) & 3;
            int ni = ei + di[d], nj = ej + dj[d];
            int t = tiles[ni][nj], ti = t / N, tj = t % N;
            md += abs(ei - ti) + abs(ej - tj) - abs(ni - ti) - abs(nj - tj);
            std::swap(tiles[ei][ej], tiles[ni][nj]);
            ei = ni; ej = nj;
            cmds.pop_back();
        }

        void move(int d, int diff) {
            std::swap(tiles[ei][ej], tiles[ei + di[d]][ej + dj[d]]);
            ei += di[d]; ej += dj[d];
            md += diff;
            cmds += d2c[d];
        }

        void move_random() {
            vector<pii> cands;
            for (int d = 0; d < 4; d++) {
                auto res = can_move(d);
                if (res.first) {
                    cands.emplace_back(d, res.second);
                }
            }
            auto [d, diff] = cands[rnd.next_int(cands.size())];
            move(d, diff);
        }

        int calc_md_naive() const {
            int res = 0;
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    int t = tiles[i][j];
                    if (t == N * N - 1) continue;
                    int ti = t / N, tj = t % N;
                    res += abs(i - ti) + abs(j - tj);
                }
            }
            return res;
        }

        bool is_solvable() const {
            // 転倒数の偶奇と空マスの偶奇が等しければ解ける
            int inv = 0;
            for (int i = 0; i < N * N - 1; i++) {
                for (int j = i + 1; j < N * N; j++) {
                    inv += tiles[i / N][i % N] > tiles[j / N][j % N];
                }
            }
            int dist = abs(ei - (N - 1)) + abs(ej - (N - 1));
            return inv % 2 == dist % 2;
        }

        void run() {
            for (int layer = 0; layer < N - 2; layer++) {
                align(layer);
            }
            move_number(IJ(N - 2, N - 2), N - 2, N - 2);
            move_empty_cell(N - 1, N - 1);
        }

        void partial_run(int layer) {
            for (int l = 0; l < layer; l++) align(l);
        }

        void run_with_beam_search(int level, double duration) {
            // level*level 以下の正方形になったら beam search を走らせる
            int align_layer_size = std::max(0, N - level);
            for (int layer = 0; layer < align_layer_size; layer++) {
                align(layer);
            }

            int offset = -align_layer_size;
            vector<std::tuple<int, int, int, int>> assign;
            for (int i = align_layer_size; i < N; i++) {
                for (int j = align_layer_size; j < N; j++) {
                    int ti = tiles[i][j] / N, tj = tiles[i][j] % N;
                    assign.emplace_back(i + offset, j + offset, ti + offset, tj + offset);
                }
            }

            NBeam::State bs(N - align_layer_size, NBeam::max_beam_turn, assign);
            int beam_width = 5000;
            if (N == 6) beam_width = 15000;
            if (N == 7) beam_width = 7000;
            bs = NBeam::beam_search(bs, beam_width, duration);

            cmds += bs.get_cmd();
        }

        static State create(int N, int seed, int nshuffle) {
            State puz;
            puz.rnd.set_seed(seed);
            puz.N = N;
            puz.ei = N - 1; puz.ej = N - 1;
            puz.tiles.resize(N, vector<int>(N));
            puz.fixed.resize(N, vector<bool>(N));
            puz.md = 0;
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    puz.tiles[i][j] = i * N + j;
                }
            }
            for (int i = 0; i < nshuffle; i++) {
                puz.move_random();
                puz.cmds.pop_back();
            }
            return puz;
        }

    };

}

#ifdef HAVE_OPENCV_HIGHGUI
void show(const vector<vector<int>>& tiles, int delay = 0) {
    int N = tiles.size();
    int sz = 800 / N, H = sz * N, W = sz * N;
    cv::Mat_<cv::Vec3b> img(H, W, cv::Vec3b(255, 255, 255));
    for (int i = 0; i <= N; i++) {
        cv::line(img, cv::Point(0, i * sz), cv::Point(W, i * sz), cv::Scalar(200, 200, 200), 2);
        cv::line(img, cv::Point(i * sz, 0), cv::Point(i * sz, H), cv::Scalar(200, 200, 200), 2);
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cv::Point ctr(j * sz + sz / 2, i * sz + sz / 2);
            for (int d = 0; d < 4; d++) if (tiles[i][j] >> d & 1) {
                cv::Point dv(dj[d] * sz / 2, di[d] * sz / 2);
                cv::line(img, ctr, ctr + dv, cv::Scalar(0, 0, 0), 10);
            }
        }
    }
    cv::imshow("img", img);
    cv::waitKey(delay);
}
#endif

void initialize() {
    c2d['L'] = 0; c2d['U'] = 1; c2d['R'] = 2; c2d['D'] = 3;
    for (char c = '0'; c <= '9'; c++) c2t[c] = c - '0';
    for (char c = 'a'; c <= 'f'; c++) c2t[c] = c - 'a' + 10;
}

int main(int argc, char** argv) {

    Timer timer;

#ifdef HAVE_OPENCV_HIGHGUI
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
#endif

    initialize();

    Input input;
    int seed = 4;
    if (argc > 1) {
        //int seed = atoi(argv[1]);
        //int seed = 4;
        dump(seed);
        std::ifstream ifs(format("tools/in/%04d.txt", seed));
        input = Input(ifs);
    }
    else {
        input = Input(cin);
    }

    int min_cost = INT_MAX, best_score = INT_MIN;
    NFlow::Result assign;
    string ans;
    int min_loop = INT_MAX, max_loop = INT_MIN, ctr_loop = 0;
    double avg_loop = 0;
    while (timer.elapsed_ms() < 500) {
        TreeModifier tmod(input, Input(input.N, rnd, input.N - 1, input.N - 1));
        int inner_loop = 0;
        while (tmod.cost) {
            tmod.local_search(rnd);
            inner_loop++;
        }
        if (tmod.cost) continue;
        chmin(min_loop, inner_loop);
        chmax(max_loop, inner_loop);
        ctr_loop++;
        avg_loop += inner_loop;

        auto res = NFlow::calc_assign(input, tmod);
        NPuzzle::State puz(input.N, res);
        if (puz.is_solvable()) {
            // solve puzzle: TODO 回数多いほどよさそう　要高速化
            if (false) {
                puz.run();
                int score = NJudge::compute_score(input.cvt(), puz.cmds);
                if (chmax(best_score, score)) {
                    best_score = score;
                    ans = puz.cmds;
                    dump(best_score);
                }
            }
            // for beam search
            if (true) {
                if (res.total_cost < min_cost) {
                    assign = res;
                    min_cost = res.total_cost;
                    dump(min_cost);
                }
            }
        }
    }
    avg_loop /= ctr_loop;
    dump(min_loop, max_loop, ctr_loop, avg_loop);

    if (true) {
        NPuzzle::State puz(input.N, assign);
        puz.run();
        int score = NJudge::compute_score(input.cvt(), puz.cmds);
        if (chmax(best_score, score)) {
            best_score = score;
            ans = puz.cmds;
            dump(best_score);
        }
    }

    if (true) { // hybrid
        NPuzzle::State puz(input.N, assign);
        puz.run_with_beam_search(8, 2900 - timer.elapsed_ms());
        int score = NJudge::compute_score(input.cvt(), puz.cmds);
        if (best_score < score) {
            best_score = score;
            ans = puz.cmds;
            dump(best_score);
        }
    }

    if (ans.size() > input.T) {
        ans = ans.substr(0, input.T);
    }
    cout << ans << endl;

    dump(timer.elapsed_ms(), best_score);

    return 0;
}