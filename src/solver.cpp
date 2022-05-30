#define _CRT_NONSTDC_NO_WARNINGS
#include <bits/stdc++.h>
#include <random>
#include <unordered_set>
#include <array>
//#include <atcoder/all>
//#include <boost/multiprecision/cpp_int.hpp>
//#include <boost/multiprecision/cpp_bin_float.hpp>
#ifdef _MSC_VER
#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
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

struct Input {
    int N, T;
    vector<vector<int>> tiles;
};

Input parse_input(std::istream& in) {
    int N, T;
    in >> N >> T;
    vector<string> S(N);
    in >> S;
    vector<vector<int>> tiles(N, vector<int>(N));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            tiles[i][j] = c2t[S[i][j]];
        }
    }
    return { N, T, tiles };
}

// TODO: 空欄を右下に限定しない方法
Input generate_tree(int N, Xorshift& rnd) {
    int T = 2 * N * N * N;
    vector<vector<int>> tiles(N, vector<int>(N));
    vector<std::tuple<int, int, int, int>> edges;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i + 1 < N && !(i + 1 == N - 1 && j == N - 1)) {
                edges.emplace_back(i, j, i + 1, j);
            }
            if (j + 1 < N && !(i == N - 1 && j + 1 == N - 1)) {
                edges.emplace_back(i, j, i, j + 1);
            }
        }
    }
    shuffle_vector(edges, rnd);
    UnionFind uf(N * N);
    for (const auto [i1, j1, i2, j2] : edges) {
        int u = i1 * N + j1, v = i2 * N + j2;
        if (!uf.same(u, v)) {
            uf.unite(u, v);
            if (i1 + 1 == i2) {
                tiles[i1][j1] |= DOWN;
                tiles[i2][j2] |= UP;
            }
            else {
                tiles[i1][j1] |= RIGHT;
                tiles[i2][j2] |= LEFT;
            }
        }
    }
    return { N, T, tiles };
}

struct TreeModifier {

    using Edge = std::tuple<int, int, int, int>;

    int N;
    vector<int> target_ctr; // target tile count
    vector<int> tree_ctr; // tree tile count
    vector<vector<int>> tiles;
    vector<Edge> disabled_edges;
    int cost;

    TreeModifier(const Input& input, const Input& tree) {
        N = input.N;
        target_ctr.assign(16, 0);
        tree_ctr.assign(16, 0);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                target_ctr[input.tiles[i][j]]++;
                tree_ctr[tree.tiles[i][j]]++;
            }
        }
        tiles = tree.tiles;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N - 1; j++) {
                int t1 = tiles[i][j], t2 = tiles[i][j + 1];
                if ((t1 & RIGHT) && (t2 & LEFT)) {
                }
                else if (!(i == N - 1 && j + 1 == N - 1)) {
                    disabled_edges.emplace_back(i, j, i, j + 1);
                }
            }
        }
        for (int i = 0; i < N - 1; i++) {
            for (int j = 0; j < N; j++) {
                int t1 = tiles[i][j], t2 = tiles[i + 1][j];
                if ((t1 & DOWN) && (t2 & UP)) {
                }
                else if (!(i + 1 == N - 1 && j == N - 1)) {
                    disabled_edges.emplace_back(i, j, i + 1, j);
                }
            }
        }
        cost = 0;
        for (int i = 0; i < 16; i++) {
            cost += abs(target_ctr[i] - tree_ctr[i]);
        }
        //dump(cost);
    }

    Edge choose_random_disabled_edge(Xorshift& rnd) {
        int eid = rnd.next_int(disabled_edges.size());
        std::swap(disabled_edges[eid], disabled_edges.back()); // 末端に移動しておく
        auto e = disabled_edges.back();
        return e;
    }

    void toggle_edge(int i1, int j1, int i2, int j2) {
        tree_ctr[tiles[i1][j1]]--;
        tree_ctr[tiles[i2][j2]]--;
        if (i1 == i2) {
            tiles[i1][j1] ^= RIGHT;
            tiles[i2][j2] ^= LEFT;
        }
        else {
            tiles[i1][j1] ^= DOWN;
            tiles[i2][j2] ^= UP;
        }
        tree_ctr[tiles[i1][j1]]++;
        tree_ctr[tiles[i2][j2]]++;
    }

    void local_search(Xorshift& rnd) {
        // 接続していない辺を on にする
        // 出来たサイクルの辺を一つ選択して off にする
        auto e = choose_random_disabled_edge(rnd);
        int i1 = std::get<0>(e), j1 = std::get<1>(e), i2 = std::get<2>(e), j2 = std::get<3>(e);
        // (i1,j1) -> (i2,j2) のパスを求める
        bool seen[10][10];
        //pii prev[10][10];
        int prev[10][10];
        auto path = [&]() {
            Fill(seen, false);
            Fill(prev, -1);
            fqu.reset();
            fqu.push((i1 << 4) | j1);
            seen[i1][j1] = true;
            bool found = false;
            while (!fqu.empty()) {
                int ij = fqu.pop(), i = ij >> 4, j = ij & 0xF;
                int t = tiles[i][j];
                for (int d = 0; d < 4; d++) {
                    if (!(t >> d & 1)) continue; // d 方向に伸びていない
                    int ni = i + di[d], nj = j + dj[d];
                    if (seen[ni][nj]) continue;
                    seen[ni][nj] = true;
                    fqu.push((ni << 4) | nj);
                    prev[ni][nj] = (i << 4) | j;
                    if (ni == i2 && nj == j2) {
                        found = true;
                        break;
                    }
                }
                if (found) break;
            }
            int p = (i2 << 4) | j2;
            vector<pii> path; path.reserve(128);
            path.emplace_back(p >> 4, p & 0xF);
            while (p != -1) {
                p = prev[p >> 4][p & 0xF];
                if (p == -1) break;
                path.emplace_back(p >> 4, p & 0xF);
            }
            return path;
        }();

        vector<Edge> cands; cands.reserve(path.size());
        for (int n = 0; n + 1 < path.size(); n++) {
            auto [i3, j3] = path[n];
            auto [i4, j4] = path[n + 1];
            if ((i3 == i4 && j3 > j4) || (j3 == j4 && i3 > i4)) {
                std::swap(i3, i4);
                std::swap(j3, j4);
            }
            cands.emplace_back(i3, j3, i4, j4);
        }

        int min_cost = INT_MAX;
        Edge min_edge;
        // connect (i1,j1)-(i2,j2)
        toggle_edge(i1, j1, i2, j2);
        for (auto e2 : cands) {
            auto [i3, j3, i4, j4] = e2;
            // disconnect (i3,j3)-(i4,j4)
            toggle_edge(i3, j3, i4, j4);
            int new_cost = 0;
            for (int t = 0; t < 16; t++) {
                new_cost += abs(target_ctr[t] - tree_ctr[t]);
            }

            int diff = new_cost - cost;
            double temp = 0.3;
            double prob = exp(-diff / temp);

            if (rnd.next_double() < prob) {
                disabled_edges.pop_back();
                disabled_edges.emplace_back(i3, j3, i4, j4);
                cost = new_cost;
                return;
            }

            if (chmin(min_cost, new_cost)) {
                min_edge = e2;
            }
            toggle_edge(i3, j3, i4, j4);
        }

        toggle_edge(i1, j1, i2, j2); // revert
    }

};

namespace NFlow {

    struct PrimalDual {
        const int INF;

        struct edge {
            int to;
            int cap;
            int cost;
            int rev;
            bool isrev;
            edge(int to = -1, int cap = -1, int cost = -1, int rev = -1, bool isrev = false)
                : to(to), cap(cap), cost(cost), rev(rev), isrev(isrev) {}
        };

        vector<vector<edge>> graph;
        vector<int> potential, min_cost;
        vector<int> prevv, preve;

        PrimalDual(int V) : INF(std::numeric_limits<int>::max()), graph(V) {}

        void add_edge(int from, int to, int cap, int cost) {
            graph[from].emplace_back(to, cap, cost, (int)graph[to].size(), false);
            graph[to].emplace_back(from, 0, -cost, (int)graph[from].size() - 1, true);
        }

        int min_cost_flow(int s, int t, int f) {
            int V = (int)graph.size();
            int ret = 0;
            using Pi = ll;
            std::priority_queue<Pi, vector<Pi>, std::greater<Pi>> que;
            potential.assign(V, 0);
            preve.assign(V, -1);
            prevv.assign(V, -1);

            while (f > 0) {
                min_cost.assign(V, INF);
                que.emplace(s);
                min_cost[s] = 0;
                while (!que.empty()) {
                    Pi p = que.top(); que.pop();
                    int pf = p >> 32, ps = p & 0xFFFFFFFFLL;
                    if (min_cost[ps] < pf) continue;
                    for (int i = 0; i < (int)graph[ps].size(); i++) {
                        edge& e = graph[ps][i];
                        int nextCost = min_cost[ps] + e.cost + potential[ps] - potential[e.to];
                        if (e.cap > 0 && min_cost[e.to] > nextCost) {
                            min_cost[e.to] = nextCost;
                            prevv[e.to] = ps, preve[e.to] = i;
                            que.emplace(((ll)min_cost[e.to] << 32) | e.to);
                        }
                    }
                }
                if (min_cost[t] == INF) return -1;
                for (int v = 0; v < V; v++) potential[v] += min_cost[v];
                int addflow = f;
                for (int v = t; v != s; v = prevv[v]) {
                    addflow = std::min(addflow, graph[prevv[v]][preve[v]].cap);
                }
                f -= addflow;
                ret += addflow * potential[t];
                for (int v = t; v != s; v = prevv[v]) {
                    edge& e = graph[prevv[v]][preve[v]];
                    e.cap -= addflow;
                    graph[v][e.rev].cap += addflow;
                }
            }
            return ret;
        }

        void output() {
            for (int i = 0; i < (int)graph.size(); i++) {
                for (auto& e : graph[i]) {
                    if (e.isrev) continue;
                    auto& rev_e = graph[e.to][e.rev];
                    cout << i << "->" << e.to << " (flow: " << rev_e.cap << "/" << rev_e.cap + e.cap << ")" << endl;
                }
            }
        }
    };

    struct Node {
        int id, i, j;
        Node(int id = -1, int i = -1, int j = -1) : id(id), i(i), j(j) {}
        std::string str() const {
            return format("Node [id=%d, p=(%d, %d)]", id, i, j);
        }
        friend std::ostream& operator<<(std::ostream& o, const Node& obj) {
            o << obj.str();
            return o;
        }
    };

    using Assign = std::pair<Node, Node>;

    struct Result {
        int total_cost;
        vector<vector<Assign>> type_to_assign;
    };

    vector<Assign> get_assign(
        const vector<Node>& S, const vector<Node>& T, const PrimalDual& pd) {
        int ns = S.size();
        vector<Assign> assign;
        for (int u = 1; u <= (int)S.size(); u++) {
            for (const auto& e : pd.graph[u]) {
                if (e.isrev) continue;
                const auto& rev_e = pd.graph[e.to][e.rev];
                if (!rev_e.cap) continue;
                // i -> e.to
                int v = e.to;
                assign.emplace_back(S[u - 1], T[v - ns - 1]);
            }
        }
        return assign;
    }

    Result calc_assign(const vector<vector<int>>& src, const vector<vector<int>>& dst) {
        Result result;
        result.type_to_assign.resize(16);
        // create nodes
        int N = src.size();
        vector<vector<Node>> S(16), T(16);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                int sc = src[i][j];
                S[sc].emplace_back(S[sc].size() + 1, i, j);
            }
        }
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                int tc = dst[i][j];
                T[tc].emplace_back(S[tc].size() + T[tc].size() + 1, i, j);
            }
        }
        // mincostflow
        int total_cost = 0;
        for (int t = 0; t < 16; t++) {
            if (S[t].empty()) continue;
            int ns = S[t].size(), nt = T[t].size();
            int V = ns + nt + 2;
            PrimalDual pd(V);
            // u=0 から v in 1..s に容量 1, コスト 0 の辺を張る
            for (const auto& v : S[t]) {
                pd.add_edge(0, v.id, 1, 0);
            }
            // u in 1..s から v in s+1...s+t に容量 inf, コスト dist(u, v) の辺を張る
            for (const auto& u : S[t]) {
                for (const auto& v : T[t]) {
                    int dist = abs(u.i - v.i) + abs(u.j - v.j);
                    pd.add_edge(u.id, v.id, pd.INF, dist);
                }
            }
            // u in s+1...s+t から v=s+t+1 に容量 1, コスト 0 の辺を張る
            for (const auto& v : T[t]) {
                pd.add_edge(v.id, V - 1, 1, 0);
            }
            //double elapsed = timer.elapsedMs();
            int cost = pd.min_cost_flow(0, V - 1, ns);
            //dump(c, cost, V, timer.elapsedMs() - elapsed);
            total_cost += cost;

            result.type_to_assign[t] = get_assign(S[t], T[t], pd);
        }

        result.total_cost = total_cost;

        return result;
    }

}


namespace NBeam2 {

    // 1-indexed にして out of bound 判定を速くする
    // 省メモリ
    // 12 x 16

    constexpr int dij[] = { -1, -16, 1, 16 }; // LURD

    uint8_t g_N, g_T;
    bool g_oob[192]; // out of bound
    uint64_t g_hash_table[192][192];

    inline int IJ(int i, int j) { return (i << 4) | j; }
    inline int I(int ij) { return ij >> 4; }
    inline int J(int ij) { return ij & 0xF; }

    void setup(const Input& input) {

        g_N = input.N;
        g_T = input.T;

        std::mt19937_64 engine;
        for (int ij = 0; ij < 192; ij++) {
            for (int n = 0; n < 192; n++) {
                g_hash_table[ij][n] = engine();
            }
        }

        Fill(g_oob, true);
        for (int i = 1; i <= g_N; i++) {
            for (int j = 1; j <= g_N; j++) {
                g_oob[IJ(i, j)] = false;
            }
        }

    }

    struct State;
    using StatePtr = std::shared_ptr<State>;
    struct State {

        int16_t turn, cost;
        uint8_t tiles[192];
        uint8_t eij;
        int8_t pdir;
        uint8_t cmds[512]; // >= 2bit * 2000
        uint64_t hash;

        State(const Input& input, const NFlow::Result& assign) {

            turn = cost = 0;
            Fill(tiles, -1);
            eij = -1;
            pdir = -1;
            Fill(cmds, 0);
            hash = 0;

            int tij_empty = IJ(input.N, input.N);
            for (const auto& as : assign.type_to_assign) {
                for (const auto& a : as) {
                    int si = a.first.i + 1, sj = a.first.j + 1, sij = IJ(si, sj); // to 1-indexed
                    int ti = a.second.i + 1, tj = a.second.j + 1, tij = IJ(ti, tj);
                    tiles[sij] = tij;
                    if (tij == tij_empty) {
                        eij = sij;
                        continue;
                    }
                    cost += cell_cost(sij);
                }
            }
            dump(cost);
        }

        inline int cell_cost(int ij) const {
            int tij = tiles[ij];
            return abs(I(ij) - I(tij)) + abs(J(ij) - J(tij));
        }

        inline bool can_move(int dir) const {
            return !(((dir + 2) & 3) == pdir || g_oob[eij + dij[dir]]);
        }

        inline void move(int dir) {
            int nij = eij + dij[dir];
            cost -= cell_cost(nij);
            hash ^= g_hash_table[eij][tiles[eij]]; hash ^= g_hash_table[nij][tiles[nij]];
            std::swap(tiles[eij], tiles[nij]);
            hash ^= g_hash_table[eij][tiles[eij]]; hash ^= g_hash_table[nij][tiles[nij]];
            cost += cell_cost(eij);
            eij = nij;
            pdir = dir;
            // 1 byte に 4 個
            cmds[turn >> 2] ^= dir << ((turn & 3) << 1);
            turn++;
        }

        void add_next_states(vector<StatePtr>& dst, std::unordered_set<uint64_t>& seen) {
            for (int d = 0; d < 4; d++) {
                if (!can_move(d)) continue;
                auto ns = std::make_shared<State>(*this);
                ns->move(d);
                if (!seen.count(ns->hash)) {
                    dst.push_back(ns);
                    seen.insert(ns->hash);
                }
            }
        }

        string get_cmd() const {
            string res;
            for (int t = 0; t < turn; t++) {
                res += d2c[(cmds[t >> 2] >> ((t & 3) << 1)) & 3];
            }
            return res;
        }

    };

    StatePtr beam_search(StatePtr init_state, double duration) {
        Timer timer;
        StatePtr best_state = init_state;
        vector<StatePtr> now_states({ init_state });
        std::unordered_set<unsigned long long> seen; seen.reserve(10000000);
        seen.insert(init_state->hash);
        int width = 30000, turn = 0;
        while (!now_states.empty() && turn < g_T && timer.elapsed_ms() < duration && best_state->cost) {
            vector<StatePtr> next_states; next_states.reserve(100000);
            for (int n = 0; n < std::min(width, (int)now_states.size()); n++) {
                now_states[n]->add_next_states(next_states, seen);
            }
            if (next_states.empty()) break;
            sort(next_states.begin(), next_states.end(), [](StatePtr& a, StatePtr& b) { return a->cost < b->cost; });
            now_states = next_states;
            if (now_states.front()->cost < best_state->cost) {
                best_state = now_states.front();
                //cerr << best_state->cost << ": " << timer.elapsed_ms() << endl;
                dump(turn, best_state->cost);
            }
            turn++;
        }
        dump(seen.size());
        cerr << best_state->get_cmd() << endl;
        return best_state;
    }

}



struct PuzzleSolver {

    int N;
    int ei, ej;
    vector<vector<int>> tiles;
    vector<vector<bool>> fixed;
    string cmds;

    // 4x4 の場合
    // 0 を (0,0) に移動
    // 1 を (0,1) に移動
    // 2 を (0,2) に移動
    // 3 を (0+2,3) に移動
    // 空きマスを (0+1,2) に移動
    // URDLURDDLUURD (2,3 が揃う)
    // 4 を (1,0) に移動
    // 8 を (2,0) に移動
    // 12 を (3,0+2) に移動
    // 空きマスを (2,0+1) に移動
    // LDRULDRRULLDR (8,12 が揃う)
    // ...

    PuzzleSolver(int N, const NFlow::Result& assign) : N(N) {
        tiles.resize(N, vector<int>(N));
        fixed.resize(N, vector<bool>(N));
        for (const auto& as : assign.type_to_assign) {
            for (const auto& a : as) {
                tiles[a.first.i][a.first.j] = a.second.i * N + a.second.j;
            }
        }
        std::tie(ei, ej) = get_pos(N * N - 1);
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

    inline int IJ(int i, int j) const {
        return i * N + j;
    }

    void align(int layer) {

        vector<std::pair<int, bool>> pib; // (target num, 特殊処理?)
        pib.emplace_back(IJ(layer, layer), false);
        for (int j = layer + 1; j < N - 1; j++) {
            pib.emplace_back(IJ(layer, j), false);
        }
        pib.emplace_back(IJ(layer, N - 1), true);
        for (int i = layer + 1; i < N - 1; i++) {
            pib.emplace_back(IJ(i, layer), false);
        }
        pib.emplace_back(IJ(N - 1, layer), true);

        for (auto [n, f] : pib) {
            int ti = n / N, tj = n % N; // 目的地
            if (f) {
                auto [si, sj] = get_pos(n);
                int dist = abs(si - ti) + abs(sj - tj);
                // 特殊処理
                if (dist == 0) {
                    // 既に揃っている
                    fixed[ti][tj] = true;
                }
                else if (tiles[ti][tj] == N * N - 1 && dist == 1) {
                    // 目的地が空マスで、距離が 1
                    int dir = get_dir(ti, tj, si, sj);
                    move(d2c[dir]);
                    fixed[ti][tj] = true;
                }
                else {
                    if (tj == N - 1) {
                        move_number(n, ti + 2, tj);
                        move_empty_cell(ti + 1, tj - 1);
                        fixed[ti][tj - 1] = fixed[ti + 2][tj] = false;
                        move("URDLURDDLUURD");
                        fixed[ti][tj - 1] = fixed[ti][tj] = true;
                    }
                    else {
                        move_number(n, ti, tj + 2);
                        move_empty_cell(ti - 1, tj + 1);
                        fixed[ti - 1][tj] = fixed[ti][tj + 2] = false;
                        move("LDRULDRRULLDR");
                        fixed[ti - 1][tj] = fixed[ti][tj] = true;
                    }

                }
            }
            else {
                move_number(n, ti, tj);
            }
        }

    }

    inline void move(char c) {
        int d = c2d[c];
        std::swap(tiles[ei][ej], tiles[ei + di[d]][ej + dj[d]]);
        ei += di[d]; ej += dj[d];
        if (!cmds.empty() && ((c2d[c] + 2) & 3) == c2d[cmds.back()]) {
            cmds.pop_back();
        }
        else {
            cmds += c;
        }
        //show();
    }

    inline void move(const string& s) {
        for (char c : s) move(c);
    }

    vector<pii> calc_zigzag_shortest_path(int si, int sj, int ti, int tj) const {
        assert(!(si == ti && sj == tj));
        bool seen[10][10];
        pii prev[10][10];
        Fill(seen, false);
        Fill(prev, pii(-1, -1));
        // TODO: abs(ti-si), abs(tj-sj) の大小で最初の移動方向を変化させる
        std::queue<std::tuple<int, int, int>> qu({ {si, sj, -1} }); // (i,j,prev_dir)
        seen[si][sj] = true;
        while (!qu.empty()) {
            auto [i, j, pd] = qu.front(); qu.pop();
            if (i == ti && j == tj) break;
            for (int d = 0; d < 4; d++) if (d != pd) {
                int ni = i + di[d], nj = j + dj[d];
                if (ni < 0 || ni >= N || nj < 0 || nj >= N || fixed[ni][nj] || seen[ni][nj]) continue;
                seen[ni][nj] = true;
                prev[ni][nj] = { i, j };
                qu.emplace(ni, nj, d);
            }
            if (pd != -1) {
                int ni = i + di[pd], nj = j + dj[pd];
                if (ni < 0 || ni >= N || nj < 0 || nj >= N || fixed[ni][nj] || seen[ni][nj]) continue;
                seen[ni][nj] = true;
                prev[ni][nj] = { i, j };
                qu.emplace(ni, nj, pd);
            }
        }
        assert(prev[ti][tj].first != -1);
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
        return path;
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
            move(d2c[dir]);
        }
    }

    int get_dir(int si, int sj, int ti, int tj) const {
        if (si == ti) {
            return sj < tj ? c2d['R'] : c2d['L'];
        }
        return si < ti ? c2d['D'] : c2d['U'];
    }

    void move_number(int n, int ti, int tj) {

        // 数字 n を (ti, tj) まで移動させる
        // n から (ti, tj) までなるべく蛇行しながら到達するルートを求める (fixed を避ける)

        auto [si, sj] = get_pos(n);

        if (si == ti && sj == tj) {
            fixed[ti][tj] = true;
            return;
        }

        auto path = calc_zigzag_shortest_path(si, sj, ti, tj);

        for (int k = 0; k + 1 < path.size(); k++) {
            auto [i1, j1] = path[k];
            auto [i2, j2] = path[k + 1];
            // 空マスを (i2,j2) に移動させる
            fixed[i1][j1] = true;
            move_empty_cell(i2, j2);
            fixed[i1][j1] = false;
            int dir = get_dir(i2, j2, i1, j1);
            move(d2c[dir]);
        }

        fixed[ti][tj] = true;
    }

#ifdef HAVE_OPENCV_HIGHGUI
    void show(int delay = 0) {
        int N = tiles.size();
        int sz = 800 / N, H = sz * N, W = sz * N;
        cv::Mat_<cv::Vec3b> img(H, W, cv::Vec3b(255, 255, 255));
        for (int i = 0; i <= N; i++) {
            cv::line(img, cv::Point(0, i * sz), cv::Point(W, i * sz), cv::Scalar(200, 200, 200), 2);
            cv::line(img, cv::Point(i * sz, 0), cv::Point(i * sz, H), cv::Scalar(200, 200, 200), 2);
        }
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (tiles[i][j] == N * N - 1) continue;
                cv::Point ctr(j * sz + sz / 3, i * sz + sz / 2);
                cv::putText(img, std::to_string(tiles[i][j]), ctr, cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);
            }
        }
        cv::imshow("img", img);
        cv::waitKey(delay);
    }
#endif

};



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

int main(int argc, char** argv) {

    Timer timer;

#ifdef HAVE_OPENCV_HIGHGUI
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
#endif

#ifdef _MSC_VER
    std::ifstream ifs("tools/in/0000.txt");
    std::istream& cin = ifs;
#endif

    c2d['L'] = 0; c2d['U'] = 1; c2d['R'] = 2; c2d['D'] = 3;
    for (char c = '0'; c <= '9'; c++) c2t[c] = c - '0';
    for (char c = 'a'; c <= 'f'; c++) c2t[c] = c - 'a' + 10;

    auto input = parse_input(cin);

    //TreeModifier tmod(input, generate_tree(input.N, rnd));

#if 0
    int min_cost = INT_MAX, loop = 0;
    string ans;
    while (timer.elapsed_ms() < 2900) {
        TreeModifier tmod(input, generate_tree(input.N, rnd));
        while (tmod.cost) {
            tmod.local_search(rnd);
        }
        if (!tmod.cost) {
            auto res = NFlow::calc_assign(input.tiles, tmod.tiles);
            PuzzleSolver puz(input.N, res);
            if (puz.is_solvable()) {
                //dump("found!", loop);
                puz.run();
                if (chmin(min_cost, (int)puz.cmds.size())) {
                    ans = puz.cmds;
                    dump(min_cost);
                }
                {
                    NBeam::StatePtr bs = std::make_shared<NBeam::State>(input, res);
                    bs = NBeam::beam_search(bs, 90000);
                    exit(1);
                }
                loop++;
            }
        }
    }
    dump(loop);
#else
    int min_cost = INT_MAX, loop = 0;
    NFlow::Result assign;
    string ans;
    while (timer.elapsed_ms() < 900) {
        loop++;
        TreeModifier tmod(input, generate_tree(input.N, rnd));
        while (tmod.cost) {
            tmod.local_search(rnd);
        }
        auto res = NFlow::calc_assign(input.tiles, tmod.tiles);
        if (res.total_cost < min_cost && PuzzleSolver(input.N, res).is_solvable()) {
            assign = res;
            min_cost = res.total_cost;
            dump(min_cost);
        }
    }
    dump(loop);

    {
        NBeam2::setup(input);
        NBeam2::StatePtr bs2 = std::make_shared<NBeam2::State>(input, assign);
        bs2 = NBeam2::beam_search(bs2, INT_MAX);
        ans = bs2->get_cmd();
    }
#endif

    if (ans.size() > input.T) {
        ans = ans.substr(0, input.T);
    }

    dump(ans.size(), double(ans.size()) / input.T);
    cout << ans << endl;

    return 0;
}