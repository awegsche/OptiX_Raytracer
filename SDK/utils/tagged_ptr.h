
template<typename... Ts>
struct TypePack {
    static constexpr size_t count = sizeof...(Ts);
};

template <typename... Ts>
class TaggedPointer {
    public:
        using Types = TypePack<Ts...>;

    private:
        static constexpr int tagShift = 57;
        static constexpr int tagBits = 64 - tagShift;
};
