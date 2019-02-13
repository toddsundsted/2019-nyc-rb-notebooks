class Dataset

  def initialize(x, y, batch_size: 5, shuffle: true)
    raise ArgumentError.new('lengths must be equal') unless x.shape[0] == y.shape[0]
    @length = x.shape[0]
    @batch_size = batch_size
    @shuffle = shuffle
    @x = x
    @y = y
  end

  def length
    @length
  end

  def each
    return enum_for unless block_given?
    x = @x
    y = MXNet::NDArray.expand_dims(@y, axis: 0).reshape([@length, 1])
    t = MXNet::NDArray.concat(x, y)
    t = MXNet::NDArray.shuffle(t) if @shuffle
    (@length/@batch_size).times do |i|
      a = t[(i*@batch_size)...((i+1)*@batch_size)]
      yield [a[0..-1, 0..1], a[0..-1, 2]]
    end
  end

end
