

const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${process.env.API_URL || 'http://localhost:8001'}/api/:path*`,
      },
    ]
  },
}

export default nextConfig
