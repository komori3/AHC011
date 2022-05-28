#define _CRT_NONSTDC_NO_WARNINGS
#include <bits/stdc++.h>
#include <random>
#include <unordered_set>
#include <array>
//#include <atcoder/all>
//#include <boost/multiprecision/cpp_int.hpp>
//#include <boost/multiprecision/cpp_bin_float.hpp>
#ifdef _MSC_VER
#include <conio.h>
#include <ppl.h>
#include <filesystem>
#include <intrin.h>
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
#else
#pragma GCC target("avx2")
#pragma GCC optimize("O3")
#pragma GCC optimize("unroll-loops")
#endif

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

    int size(int k) {
        return -data[find(k)];
    }

    bool same(int x, int y) {
        return find(x) == find(y);
    }
};



struct State;
using StatePtr = std::shared_ptr<State>;
struct State {

    const int N, T;

    int turn;               // ターン数
    int si, sj;             // 空きマスの位置
    int pdir;               // 前回の移動方向
    vector<string> board;   // 盤面
    string cmds;            // コマンド列
    int score;              // スコア

    State(int N, const vector<string>& board) : N(N), T(N* N* N * 2), turn(0), si(-1), sj(-1), pdir(-1), board(board) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (board[i][j] == '0') {
                    si = i;
                    sj = j;
                }
            }
        }
        score = calc_score();
    }

    inline bool can_move(int dir) const {
        if (((dir + 2) & 3) == pdir) return false; // 往復は考えなくてよい
        int ni = si + di[dir], nj = sj + dj[dir];
        return 0 <= ni && ni < N && 0 <= nj && nj < N;
    }

    void move(int dir) {
        turn++;
        std::swap(board[si][sj], board[si + di[dir]][sj + dj[dir]]);
        si += di[dir]; sj += dj[dir];
        pdir = dir;
        cmds += d2c[dir];
        score = calc_score();
    }

    int calc_score() const {
        UnionFind uf(N * N);
        vector<bool> tree(N * N, true);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (i + 1 < N && (c2t[board[i][j]] & DOWN) && (c2t[board[i + 1][j]] & UP)) {
                    int a = uf.find(i * N + j), b = uf.find((i + 1) * N + j);
                    if (a == b) {
                        tree[a] = false;
                    }
                    else {
                        bool t = tree[a] && tree[b];
                        uf.unite(a, b);
                        tree[uf.find(a)] = t;
                    }
                }
                if (j + 1 < N && (c2t[board[i][j]] & RIGHT) && (c2t[board[i][j + 1]] & LEFT)) {
                    int a = uf.find(i * N + j), b = uf.find(i * N + (j + 1));
                    if (a == b) {
                        tree[a] = false;
                    }
                    else {
                        bool t = tree[a] && tree[b];
                        uf.unite(a, b);
                        tree[uf.find(a)] = t;
                    }
                }
            }
        }
        int max_tree = -1;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (c2t[board[i][j]] && tree[uf.find(i * N + j)]) {
                    if (max_tree == -1 || uf.size(max_tree) < uf.size(i * N + j)) {
                        max_tree = i * N + j;
                    }
                }
            }
        }
        return max_tree == -1 ? 0 : uf.size(max_tree);
    }

    void add_next_states(vector<StatePtr>& dst) {
        for (int d = 0; d < 4; d++) {
            if (!can_move(d)) continue;
            auto ns = std::make_shared<State>(*this);
            ns->move(d);
            dst.push_back(ns);
        }
    }

};

StatePtr beam_search(StatePtr init_state, double duration) {
    Timer timer;
    StatePtr best_state = init_state;
    vector<StatePtr> now_states({ init_state });
    int width = 5000, turn = 0;
    while (!now_states.empty() && turn < init_state->T && timer.elapsed_ms() < duration) {
        vector<StatePtr> next_states;
        for (int n = 0; n < std::min(width, (int)now_states.size()); n++) {
            now_states[n]->add_next_states(next_states);
        }
        if (next_states.empty()) break;
        sort(next_states.begin(), next_states.end(), [](StatePtr& a, StatePtr& b) { return a->score > b->score; });
        now_states = next_states;
        if (best_state->score < now_states.front()->score) {
            best_state = now_states.front();
            cerr << best_state->score << ": " << timer.elapsed_ms() << endl;
        }
        turn++;
    }
    return best_state;
}

int main(int argc, char** argv) {

#ifdef HAVE_OPENCV_HIGHGUI
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
#endif

    c2d['L'] = 0; c2d['U'] = 1; c2d['R'] = 2; c2d['D'] = 3;
    for (char c = '0'; c <= '9'; c++) c2t[c] = c - '0';
    for (char c = 'a'; c <= 'f'; c++) c2t[c] = c - 'a' + 10;

    StatePtr init_state = nullptr;
    {
        int N, T;
        cin >> N >> T;
        vector<string> board(N);
        cin >> board;
        init_state = std::make_shared<State>(N, board);
    }

    auto best_state = beam_search(init_state, 2900);

    cout << best_state->cmds << endl;

    return 0;
}